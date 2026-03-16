#include "BinSorterRenderer.h"
#include "VideoAssetPool.h"
#include "ofMain.h"
#include <algorithm>

static void buildSlotsImpl(BinSorter* binSorter, VideoAssetPool* videoPool, bool videoLoop,
    const std::vector<std::vector<BinItem>>& bins,
    const std::map<std::pair<int, int>, NestedBinData>& nestedBins,
    std::vector<VideoSlot>& out, bool quiet) {
    out.clear();
    if (!binSorter) return;

    int boxW = binSorter->getBoxWidth();
    const int padding = 20;

    for (size_t bi = 0; bi < bins.size(); ++bi) {
        float baseX = bi * (boxW + padding);
        float baseY = 0;

        for (const auto& it : bins[bi]) {
            auto itNested = nestedBins.find({(int)bi, it.itemIdx});
            if (itNested != nestedBins.end()) {
                for (const auto& nit : itNested->second.items) {
                    int wr = 0, hr = 0;
                    binSorter->getItemRatio(nit.w, nit.h, wr, hr);
                    std::string path = videoPool ? videoPool->getVideoPath(wr, hr) : "";
                    std::string nextPath = (!videoLoop && videoPool) ? videoPool->getVideoPath(wr, hr) : "";
                    VideoSlot slot;
                    slot.x = (int)(baseX + it.x + nit.x);
                    slot.y = (int)(baseY + it.y + nit.y);
                    slot.w = nit.w;
                    slot.h = nit.h;
                    slot.ratioW = wr;
                    slot.ratioH = hr;
                    slot.path = path;
                    slot.nextPath = nextPath;
                    if (!path.empty()) {
                        if (slot.player.load(path)) {
                            slot.player.setLoopState(videoLoop ? OF_LOOP_NORMAL : OF_LOOP_NONE);
                            slot.player.play();
                            if (!videoLoop && !nextPath.empty() && slot.nextPlayer.load(nextPath)) {
                                slot.nextPlayer.setLoopState(OF_LOOP_NONE);
                                slot.nextPlayer.setPosition(0);
                            }
                            slot.hasVideo = true;
                        } else {
                            slot.hasVideo = false;
                            if (!quiet) ofLogWarning("BinSorterRenderer") << "load failed for ratio " << wr << "_" << hr << ": " << path;
                        }
                    } else {
                        slot.hasVideo = false;
                    }
                    if (!quiet) {
                        float slotAspect = (slot.h > 0) ? (float)slot.w / slot.h : 0;
                        ofLogNotice("BinSorterRenderer") << "Slot " << (out.size() + 1) << ": "
                            << slot.w << "x" << slot.h << " aspect=" << slotAspect
                            << " -> ratio " << wr << ":" << hr << (path.empty() ? "" : " ") << path;
                    }
                    out.push_back(slot);
                }
            } else {
                int wr = 0, hr = 0;
                binSorter->getItemRatio(it.w, it.h, wr, hr);
                std::string path = videoPool ? videoPool->getVideoPath(wr, hr) : "";
                std::string nextPath = (!videoLoop && videoPool) ? videoPool->getVideoPath(wr, hr) : "";
                VideoSlot slot;
                slot.x = (int)(baseX + it.x);
                slot.y = (int)(baseY + it.y);
                slot.w = it.w;
                slot.h = it.h;
                slot.ratioW = wr;
                slot.ratioH = hr;
                slot.path = path;
                slot.nextPath = nextPath;
                if (!path.empty()) {
                    if (slot.player.load(path)) {
                        slot.player.setLoopState(videoLoop ? OF_LOOP_NORMAL : OF_LOOP_NONE);
                        slot.player.play();
                        if (!videoLoop && !nextPath.empty() && slot.nextPlayer.load(nextPath)) {
                            slot.nextPlayer.setLoopState(OF_LOOP_NONE);
                            slot.nextPlayer.setPosition(0);
                        }
                        slot.hasVideo = true;
                    } else {
                        slot.hasVideo = false;
                        if (!quiet) ofLogWarning("BinSorterRenderer") << "load failed for ratio " << wr << "_" << hr << ": " << path;
                    }
                } else {
                    slot.hasVideo = false;
                }
                if (!quiet) {
                    float slotAspect = (slot.h > 0) ? (float)slot.w / slot.h : 0;
                    ofLogNotice("BinSorterRenderer") << "Slot " << (out.size() + 1) << ": "
                        << slot.w << "x" << slot.h << " aspect=" << slotAspect
                        << " -> ratio " << wr << ":" << hr << (path.empty() ? "" : " ") << path;
                }
                out.push_back(slot);
            }
        }
    }
}

void BinSorterRenderer::setup(BinSorter* sorter, VideoAssetPool* pool, bool videoLoop_) {
    binSorter = sorter;
    videoPool = pool;
    videoLoop = videoLoop_;
    if (videoPool) videoPool->resetUsed();
    buildSlots();
}

void BinSorterRenderer::buildSlots() {
    if (!binSorter) return;
    buildSlotsFromArrangement(binSorter->getBins(), binSorter->getNestedBins(), slots);
}

void BinSorterRenderer::buildSlotsFromArrangement(const std::vector<std::vector<BinItem>>& bins,
    const std::map<std::pair<int, int>, NestedBinData>& nestedBins,
    std::vector<VideoSlot>& out) {
    buildSlotsImpl(binSorter, videoPool, videoLoop, bins, nestedBins, out, false);
}

void BinSorterRenderer::preloadFromArrangement(const Arrangement& arr) {
    if (!binSorter) return;
    nextSlots.clear();
    buildSlotsImpl(binSorter, videoPool, videoLoop, arr.bins, arr.nestedBins, nextSlots, true);
    for (auto& slot : nextSlots) {
        if (slot.hasVideo && slot.nextPlayer.isLoaded())
            slot.nextPlayer.update();
    }
}

void BinSorterRenderer::swapToPreloaded(const Arrangement& arr) {
    if (nextSlots.empty()) return;
    binSorter->loadArrangement(arr.bins, arr.nestedBins);
    std::swap(slots, nextSlots);
    nextSlots.clear();
}

void BinSorterRenderer::update() {
    for (auto& slot : nextSlots) {
        if (slot.hasVideo) {
            slot.player.update();
            if (!videoLoop && slot.nextPlayer.isLoaded())
                slot.nextPlayer.update();
        }
    }
    for (size_t i = 0; i < slots.size(); ++i) {
        auto& slot = slots[i];
        if (slot.hasVideo) {
            slot.player.update();
            if (!videoLoop) {
                if (slot.nextPlayer.isLoaded())
                    slot.nextPlayer.update();  // keep preloaded video decoded for seamless swap
                if (slot.player.getIsMovieDone()) {
                    if (slot.nextPlayer.isLoaded()) {
                        slot.nextPlayer.play();
                        slot.player.close();
                        std::string newPath = videoPool ? videoPool->getVideoPath(slot.ratioW, slot.ratioH) : "";
                        if (newPath.empty()) {
                            slot.hasVideo = false;
                            ofLogWarning("BinSorterRenderer") << "Slot " << (i + 1) << ": no replacement video for ratio "
                                << slot.ratioW << ":" << slot.ratioH << ", falling back to placeholder";
                        } else {
                            if (slot.player.load(newPath)) {
                                slot.player.setLoopState(OF_LOOP_NONE);
                                slot.player.setPosition(0);
                                slot.path = slot.nextPath;
                                slot.nextPath = newPath;
                                std::swap(slot.player, slot.nextPlayer);
                                float slotAspect = (slot.h > 0) ? (float)slot.w / slot.h : 0;
                                ofLogNotice("BinSorterRenderer") << "Slot " << (i + 1) << ": "
                                    << slot.w << "x" << slot.h << " aspect=" << slotAspect
                                    << " -> ratio " << slot.ratioW << ":" << slot.ratioH << " " << slot.path;
                            } else {
                                slot.hasVideo = false;
                                ofLogWarning("BinSorterRenderer") << "Slot " << (i + 1) << ": load failed for ratio "
                                    << slot.ratioW << "_" << slot.ratioH << ": " << newPath;
                            }
                        }
                    } else {
                        slot.player.close();
                        std::string path = videoPool ? videoPool->getVideoPath(slot.ratioW, slot.ratioH) : "";
                        if (path.empty()) {
                            slot.hasVideo = false;
                            ofLogWarning("BinSorterRenderer") << "Slot " << (i + 1) << ": no replacement video for ratio "
                                << slot.ratioW << ":" << slot.ratioH << ", falling back to placeholder";
                        } else {
                            if (slot.player.load(path)) {
                                slot.path = path;
                                slot.player.setLoopState(OF_LOOP_NONE);
                                slot.player.play();
                                float slotAspect = (slot.h > 0) ? (float)slot.w / slot.h : 0;
                                ofLogNotice("BinSorterRenderer") << "Slot " << (i + 1) << ": "
                                    << slot.w << "x" << slot.h << " aspect=" << slotAspect
                                    << " -> ratio " << slot.ratioW << ":" << slot.ratioH << " " << path;
                            } else {
                                slot.hasVideo = false;
                                ofLogWarning("BinSorterRenderer") << "Slot " << (i + 1) << ": load failed for ratio "
                                    << slot.ratioW << "_" << slot.ratioH << ": " << path;
                            }
                        }
                    }
                }
            }
        }
    }
}

void BinSorterRenderer::draw(float offsetX, float offsetY) {
    if (!binSorter) return;
    int boxW = binSorter->getBoxWidth();
    int boxH = binSorter->getBoxHeight();
    auto& bins = binSorter->getBins();
    const int padding = 20;

    // Fill area outside content with black (when window is larger than render, e.g. fullscreen)
    int winW = ofGetWindowWidth(), winH = ofGetWindowHeight();
    int contentW = (int)(bins.size() * (boxW + padding) - (bins.empty() ? 0 : padding));
    if (boxH < winH || contentW < winW) {
        ofPushView();
        ofViewport(0, 0, winW, winH, false);
        ofFill();
        ofSetColor(0);
        if (boxH < winH)
            ofDrawRectangle(0, 0, winW, winH - boxH);
        if (contentW < winW)
            ofDrawRectangle(contentW, 0, winW - contentW, winH);
        ofPopView();
    }

    for (size_t bi = 0; bi < bins.size(); ++bi) {
        float offX = offsetX + bi * (boxW + padding);
        float offY = offsetY;
        ofNoFill();
        ofSetColor(0);
        ofDrawRectangle(offX, offY, boxW, boxH);
    }

    // Draw smallest first, largest last so overlaps favor larger items
    std::vector<size_t> drawOrder(slots.size());
    for (size_t i = 0; i < slots.size(); ++i) drawOrder[i] = i;
    std::sort(drawOrder.begin(), drawOrder.end(), [this](size_t a, size_t b) {
        int areaA = slots[a].w * slots[a].h;
        int areaB = slots[b].w * slots[b].h;
        return areaA < areaB;
    });

    for (size_t idx : drawOrder) {
        const auto& slot = slots[idx];
        float dx = offsetX + slot.x;
        float dy = offsetY + slot.y;
        if (slot.hasVideo && slot.player.isLoaded()) {
            ofSetColor(255);  // white - needed so video texture isn't tinted black
            float vw = slot.player.getWidth();
            float vh = slot.player.getHeight();
            // Prefer actual texture dimensions - player metadata can differ for non-square pixels
            const ofTexture& tex = slot.player.getTexture();
            if (tex.isAllocated()) {
                float tw = tex.getWidth(), th = tex.getHeight();
                if (tw > 0 && th > 0) { vw = tw; vh = th; }
            }
            if (vw > 0 && vh > 0) {
                // Aspect-fill (cover): scale video to completely fill the slot, center and crop overflow.
                // This removes white lines when slots have been expanded by stretchToEdges.
                float slotAspect = (float)slot.w / slot.h;
                float vidAspect = vw / vh;
                float drawW, drawH, drawX, drawY;
                if (vidAspect > slotAspect) {
                    drawH = slot.h;
                    drawW = slot.h * vidAspect;
                    drawX = dx + (slot.w - drawW) * 0.5f;
                    drawY = dy;
                } else {
                    drawW = slot.w;
                    drawH = slot.w / vidAspect;
                    drawX = dx;
                    drawY = dy + (slot.h - drawH) * 0.5f;
                }
                slot.player.draw(drawX, drawY, drawW, drawH);
            } else {
                float ol = 0.5f;
                slot.player.draw(dx - ol, dy - ol, slot.w + 2*ol, slot.h + 2*ol);
            }
        } else {
            if (slot.hasVideo) {
                std::string key = slot.path + "_" + std::to_string(slot.x) + "_" + std::to_string(slot.y);
                if (loggedNotReadyKeys.find(key) == loggedNotReadyKeys.end()) {
                    loggedNotReadyKeys.insert(key);
                    ofLogWarning("BinSorterRenderer") << "video not ready yet (isLoaded=false): "
                        << slot.path << " at (" << slot.x << "," << slot.y << ")";
                }
            }
            ofFill();
            ofSetColor(100, 150, 200);
            ofDrawRectangle(dx, dy, slot.w, slot.h);
        }
    }
}

void BinSorterRenderer::drawToFbo(ofFbo& fbo) {
    fbo.begin();
    ofClear(255, 255);
    draw(0, 0);
    fbo.end();
}

void BinSorterRenderer::regenerate() {
    if (binSorter) {
        if (videoPool) videoPool->resetUsed();
        loggedNotReadyKeys.clear();
        slots.clear();
        buildSlots();
    }
}
