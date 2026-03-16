#include "ofApp.h"
#include "ArrangementIO.h"
#include <algorithm>
#include <set>
#include <random>

void ofApp::setup() {
    ofSetBackgroundColor(0, 0, 0);
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

    renderer.setup(binSorter.get(), &videoPool, config.videoLoop);

    ofSetWindowShape(config.boxWidth, config.boxHeight);
    exportFbo.allocate(config.boxWidth, config.boxHeight, GL_RGB);
    fadeContentFbo.allocate(config.boxWidth, config.boxHeight, GL_RGBA);

    if (arrangements.size() > 1) {
        nextLayoutIdx = pickNextArrangementIndex();
        renderer.preloadFromArrangement(arrangements[nextLayoutIdx]);
    }
    scheduleNextTransition();
}

void ofApp::logArrangementInfo(size_t idx) {
    if (arrangements.empty() || idx >= arrangements.size()) return;
    ofLogNotice("ofApp") << "Picked arrangement " << (idx + 1) << " of " << arrangements.size();
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

void ofApp::swapToPreloadedAndLog(size_t idx, bool deferPlay) {
    if (arrangements.empty() || idx >= arrangements.size()) return;
    if (!renderer.hasPreloadedLayout()) {
        pickAndLoadArrangement(idx);
        return;
    }
    renderer.swapToPreloaded(arrangements[idx]);
    if (!deferPlay) renderer.startPlaying();
    logArrangementInfo(idx);
}

void ofApp::preloadNextLayout() {
    if (arrangements.size() <= 1) return;
    nextLayoutIdx = pickNextArrangementIndex();
    renderer.preloadFromArrangement(arrangements[nextLayoutIdx]);
}

void ofApp::pickAndLoadArrangement(size_t idx) {
    if (arrangements.empty() || idx >= arrangements.size()) return;
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

float ofApp::scheduleNextTransition() {
    float minT = config.transitionTimerMin;
    float maxT = config.transitionTimerMax;
    if (maxT < minT) maxT = minT;
    float delay = ofRandom(minT, maxT);
    nextTransitionTime = ofGetElapsedTimef() + delay;
    return delay;
}

size_t ofApp::pickNextArrangementIndex() {
    if (arrangements.empty()) return 0;
    if (pickQueue.empty()) {
        for (size_t i = 0; i < arrangements.size(); ++i) pickQueue.push_back(i);
        std::shuffle(pickQueue.begin(), pickQueue.end(), std::mt19937(std::random_device{}()));
    }
    size_t idx = pickQueue.back();
    pickQueue.pop_back();
    return idx;
}

void ofApp::update() {
    float now = ofGetElapsedTimef();

    if (transitionState == TransitionState::Idle) {
        bool trigger = arrangementPickRequested || (now >= nextTransitionTime);
        if (trigger && !arrangements.empty()) {
            arrangementPickRequested = false;

            if (config.transitionType == TransitionType::Jumpcut) {
                swapToPreloadedAndLog(nextLayoutIdx);
                preloadNextLayout();
                float nextTimer = scheduleNextTransition();
                ofLogNotice("ofApp") << "XXXXXXXXXXX";
                ofLogNotice("ofApp") << "Transition: type=jumpcut duration=0s next_transition_timer=" << nextTimer << "s";
                ofLogNotice("ofApp") << "XXXXXXXXXXX";
            } else if (config.transitionType == TransitionType::Fade) {
                transitionState = TransitionState::FadeDown;
                transitionStartTime = now;
            } else {
                transitionState = TransitionState::HoldBlack;
                transitionStartTime = now;
            }
        }
    } else if (transitionState == TransitionState::FadeDown) {
        float dur = std::max(0.016f, config.transitionDurationFade);
        if (now - transitionStartTime >= dur) {
            transitionState = TransitionState::FadeHoldBlack;
            transitionStartTime = now;
            fadeHoldBlackSwapDone = false;
        }
    } else if (transitionState == TransitionState::FadeHoldBlack) {
        if (!fadeHoldBlackSwapDone) {
            swapToPreloadedAndLog(nextLayoutIdx, true);
            fadeHoldBlackSwapDone = true;
        } else {
            renderer.startPlaying();
            transitionState = TransitionState::FadeUp;
            transitionStartTime = now;
        }
    } else if (transitionState == TransitionState::HoldBlack) {
        float dur = std::max(0.016f, config.transitionDurationJumpToBlack);
        if (now - transitionStartTime >= dur) {
            swapToPreloadedAndLog(nextLayoutIdx);
            preloadNextLayout();
            float nextTimer = scheduleNextTransition();
            transitionState = TransitionState::Idle;
            ofLogNotice("ofApp") << "XXXXXXXXXXX";
            ofLogNotice("ofApp") << "Transition: type=jumpcut_to_black duration=" << dur << "s next_transition_timer=" << nextTimer << "s";
            ofLogNotice("ofApp") << "XXXXXXXXXXX";
        }
    } else if (transitionState == TransitionState::FadeUp) {
        float dur = std::max(0.016f, config.transitionDurationFade);
        if (now - transitionStartTime >= dur) {
            float nextTimer = scheduleNextTransition();
            transitionState = TransitionState::Idle;
            preloadNextLayout();
            float totalDur = config.transitionDurationFade * 2.f;
            ofLogNotice("ofApp") << "XXXXXXXXXXX";
            ofLogNotice("ofApp") << "Transition: type=fade duration=" << totalDur << "s next_transition_timer=" << nextTimer << "s";
            ofLogNotice("ofApp") << "XXXXXXXXXXX";
        }
    }

    renderer.update();
}

void ofApp::draw() {
    ofBackground(0);

    if (transitionState == TransitionState::HoldBlack || transitionState == TransitionState::FadeHoldBlack) {
        ofFill();
        ofSetColor(0);
        ofDrawRectangle(0, 0, ofGetWindowWidth(), ofGetWindowHeight());
    } else if (transitionState == TransitionState::FadeUp) {
        renderer.draw(0, 0);
        float dur = std::max(0.016f, config.transitionDurationFade);
        float elapsed = ofGetElapsedTimef() - transitionStartTime;
        float t = std::min(1.f, elapsed / dur);
        ofEnableAlphaBlending();
        ofFill();
        ofSetColor(0, 0, 0, (int)((1.f - t) * 255));
        ofDrawRectangle(0, 0, ofGetWindowWidth(), ofGetWindowHeight());
        ofDisableAlphaBlending();
    } else {
        renderer.draw(0, 0);

        if (transitionState == TransitionState::FadeDown) {
            float dur = std::max(0.016f, config.transitionDurationFade);
            float elapsed = ofGetElapsedTimef() - transitionStartTime;
            float t = std::min(1.f, elapsed / dur);
            ofEnableAlphaBlending();
            ofFill();
            ofSetColor(0, 0, 0, (int)(t * 255));
            ofDrawRectangle(0, 0, ofGetWindowWidth(), ofGetWindowHeight());
            ofDisableAlphaBlending();
        }
    }

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
