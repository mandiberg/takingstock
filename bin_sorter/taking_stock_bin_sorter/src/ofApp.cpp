#include "ofApp.h"
#include "ArrangementIO.h"
#include <algorithm>
#include <set>
#include <random>

void ofApp::setup() {
    if (!ConfigLoader::load("config.txt", config)) {
        ofLogError("ofApp") << "Failed to load config.txt, using defaults";
        config.sizeRatios.push_back(SizeRatio(1, 1, 0.3f, 0.2f, 0.2f));
        config.sizeRatios.push_back(SizeRatio(2, 3, 0.2f, 0.2f, 0.2f));
        config.sizeRatios.push_back(SizeRatio(3, 2, 0.2f, 0.2f, 0.2f));
    }

    if (!videoPool.load(config.videoAssetPath)) {
        ofLogWarning("ofApp") << "No video assets found, will use colored rects";
    }

    binSorter = std::make_unique<BinSorter>(config.boxWidth, config.boxHeight, config.sizeRatios,
                         config.packingStopArea, config.nestingLayers,
                         config.nestedMinSpaceThreshold, config.mainBinFillChance,
                         config.itemBreakScale, config.itemBreakChance,
                         config.breakBoxMinItems, config.breakBoxMaxItems,
                         config.breakBoxFillAttempts, config.breakBoxCoverageThreshold,
                         config.placementAreaExponent, config.placementTopK);

    std::string arrangementPath = ArrangementIO::findArrangementPath(config.arrangementsPath, config.boxWidth, config.boxHeight, config.nestingLayers);

    if (!arrangementPath.empty() && ArrangementIO::load(arrangementPath, arrangements)) {
        auto it = std::remove_if(arrangements.begin(), arrangements.end(),
            [this](const Arrangement& a) {
                if (!ArrangementIO::isValidArrangement(a, config.boxWidth, config.boxHeight))
                    return true;
                if (config.gapFilterThreshold >= 0) {
                    binSorter->loadArrangement(a.bins, a.nestedBins);
                    int maxGap = binSorter->getLargestFittableAreaInLayout();
                    int threshold = (config.gapFilterThreshold == 0) ? 1 : config.gapFilterThreshold;
                    return maxGap >= threshold;
                }
                return false;
            });
        arrangements.erase(it, arrangements.end());
        if (!arrangements.empty()) {
            ofLogNotice("ofApp") << "Loaded " << arrangements.size() << " arrangements from disk"
                << (config.gapFilterThreshold >= 0 ? " (filtered by gap threshold)" : "");
        }
        if (arrangements.empty()) {
            ofLogWarning("ofApp") << "All loaded arrangements were invalid or exceeded gap threshold, generating new";
        }
    }
    if (arrangementPath.empty() || arrangements.empty()) {
        // Precompute arrangements with multiple phases (different seeds explore different regions)
        arrangements.clear();
        std::set<std::string> seenSignatures;

        for (int phase = 0; phase < config.layoutPhases; ++phase) {
            ofSetRandomSeed((unsigned long)(phase + 1));  // different seed per phase
            int phaseStartCount = (int)arrangements.size();
            int attempts = 0;
            int staleCount = 0;

            while (attempts < config.layoutMaxAttempts) {
                binSorter->sort(-1);
                std::string sig = binSorter->getLayoutSignature();
                if (seenSignatures.insert(sig).second) {
                    Arrangement arr;
                    arr.bins = binSorter->getBins();
                    arr.nestedBins = binSorter->getNestedBins();
                    if (!ArrangementIO::isValidArrangement(arr, config.boxWidth, config.boxHeight)) {
                        staleCount++;
                        if (staleCount >= config.layoutStaleThreshold) break;
                    } else if (config.gapFilterThreshold >= 0) {
                        int maxGap = binSorter->getLargestFittableAreaInLayout();
                        int threshold = (config.gapFilterThreshold == 0) ? 1 : config.gapFilterThreshold;
                        if (maxGap >= threshold) {
                            staleCount++;
                            if (staleCount >= config.layoutStaleThreshold) break;
                        } else {
                            arrangements.push_back(arr);
                            staleCount = 0;
                            if (arrangements.size() % 10 == 0) {
                                ofLogNotice("ofApp") << "Arrangements: " << arrangements.size();
                            }
                        }
                    } else {
                        arrangements.push_back(arr);
                        staleCount = 0;
                        if (arrangements.size() % 10 == 0) {
                            ofLogNotice("ofApp") << "Arrangements: " << arrangements.size();
                        }
                    }
                } else {
                    staleCount++;
                    if (staleCount >= config.layoutStaleThreshold) break;
                }
                attempts++;
            }

            int phaseNew = (int)arrangements.size() - phaseStartCount;
            ofLogNotice("ofApp") << "Phase " << (phase + 1) << ": found " << phaseNew << " new (total " << arrangements.size() << ")";
        }

        ofLogNotice("ofApp") << "Finished: " << arrangements.size() << " unique arrangements";

        if (!arrangements.empty()) {
            std::string savePath = ArrangementIO::getArrangementPath(config.arrangementsPath, config.boxWidth, config.boxHeight, config.nestingLayers, (int)arrangements.size());
            ArrangementIO::save(savePath, arrangements);
        }
    }

    if (arrangements.empty()) {
        ofLogError("ofApp") << "No arrangements found, using single sort result";
        binSorter->sort(-1);
    } else {
        binSorter->loadArrangement(arrangements[0].bins, arrangements[0].nestedBins);
    }

    pickQueue.clear();
    for (size_t i = 0; i < arrangements.size(); ++i) pickQueue.push_back(i);
    std::shuffle(pickQueue.begin(), pickQueue.end(), std::mt19937(std::random_device{}()));

    renderer.setup(binSorter.get(), &videoPool);

    ofSetWindowShape(config.boxWidth, config.boxHeight);
    exportFbo.allocate(config.boxWidth, config.boxHeight, GL_RGB);
}

void ofApp::update() {
    if (arrangementPickRequested && !arrangements.empty()) {
        arrangementPickRequested = false;
        if (pickQueue.empty()) {
            for (size_t i = 0; i < arrangements.size(); ++i) pickQueue.push_back(i);
            std::shuffle(pickQueue.begin(), pickQueue.end(), std::mt19937(std::random_device{}()));
        }
        size_t idx = pickQueue.back();
        pickQueue.pop_back();
        binSorter->loadArrangement(arrangements[idx].bins, arrangements[idx].nestedBins);
        ofLogNotice("ofApp") << "Picked arrangement " << (idx + 1) << " of " << arrangements.size();
        renderer.regenerate();
        const auto& slots = renderer.getSlots();
        bool hasBreakBox = !arrangements[idx].nestedBins.empty();
        ofLogNotice("ofApp") << "Window: " << config.boxWidth << " x " << config.boxHeight
            << " | nestingLayers=" << config.nestingLayers
            << " | breakBox=" << (hasBreakBox ? "yes" : "no");
        if (hasBreakBox) {
            const auto& bins = arrangements[idx].bins;
            const auto& nestedBins = arrangements[idx].nestedBins;
            int bbNum = 0;
            for (const auto& kv : nestedBins) {
                int bi = kv.first.first;
                int parentIdx = kv.first.second;
                int nx = kv.second.parentX, ny = kv.second.parentY;
                int nw = kv.second.parentW, nh = kv.second.parentH;
                if (bi >= 0 && bi < (int)bins.size()) {
                    for (const auto& pit : bins[bi]) {
                        if (pit.itemIdx == parentIdx) {
                            nx = pit.x; ny = pit.y; nw = pit.w; nh = pit.h;
                            break;
                        }
                    }
                }
                int nItems = (int)kv.second.items.size();
                ofLogNotice("ofApp") << "  breakBox " << (++bbNum) << ": " << nItems << " items"
                    << " at (" << nx << "," << ny << ") size " << nw << "x" << nh;
            }
        }
        for (size_t i = 0; i < slots.size(); ++i) {
            const auto& s = slots[i];
            bool outOfBounds = (s.x + s.w > config.boxWidth || s.y + s.h > config.boxHeight ||
                                s.x < 0 || s.y < 0);
            ofLogNotice("ofApp") << "  Slot " << (i + 1) << ": x=" << s.x << " y=" << s.y
                << " w=" << s.w << " h=" << s.h << " right=" << (s.x + s.w)
                << " bottom=" << (s.y + s.h) << (outOfBounds ? " [OUT OF BOUNDS]" : "");
        }
    }
    renderer.update();
}

void ofApp::draw() {
    ofBackground(240);
    renderer.draw(0, 0);

    if (exportRequested) {
        renderer.drawToFbo(exportFbo);
        ofPixels pixels;
        exportFbo.readToPixels(pixels);
        ofImage img;
        img.setFromPixels(pixels);
        img.save("bin_sorter_export.png");
        ofLogNotice("ofApp") << "Exported to bin_sorter_export.png";
        exportRequested = false;
    }
}

void ofApp::keyPressed(int key) {
    if (key == 's' || key == 'S') {
        exportRequested = true;
    } else if (key == 'r' || key == 'R') {
        arrangementPickRequested = true;
    }
}
