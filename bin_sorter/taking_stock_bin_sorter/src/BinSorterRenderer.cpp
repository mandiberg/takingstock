#include "BinSorterRenderer.h"
#include "VideoAssetPool.h"
#include "ofMain.h"
#include <algorithm>

void BinSorterRenderer::setup(BinSorter* sorter, VideoAssetPool* pool) {
    binSorter = sorter;
    videoPool = pool;
    if (videoPool) videoPool->resetUsed();
    buildSlots();
}

void BinSorterRenderer::buildSlots() {
    slots.clear();
    if (!binSorter) return;

    auto& bins = binSorter->getBins();
    auto& nestedBins = binSorter->getNestedBins();
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
                    VideoSlot slot;
                    slot.x = (int)(baseX + it.x + nit.x);
                    slot.y = (int)(baseY + it.y + nit.y);
                    slot.w = nit.w;
                    slot.h = nit.h;
                    slot.ratioW = wr;
                    slot.ratioH = hr;
                    slot.path = path;
                    if (!path.empty()) {
                        if (slot.player.load(path)) {
                            slot.player.setLoopState(OF_LOOP_NORMAL);
                            slot.player.play();
                            slot.hasVideo = true;
                        } else {
                            slot.hasVideo = false;
                            ofLogWarning("BinSorterRenderer") << "load failed for ratio "
                                << wr << "_" << hr << ": " << path;
                        }
                    } else {
                        slot.hasVideo = false;
                    }
                    float slotAspect = (slot.h > 0) ? (float)slot.w / slot.h : 0;
                    ofLogNotice("BinSorterRenderer") << "Slot " << (slots.size() + 1) << ": "
                        << slot.w << "x" << slot.h << " aspect=" << slotAspect
                        << " -> ratio " << wr << ":" << hr << (path.empty() ? "" : " ") << path;
                    slots.push_back(slot);
                }
            } else {
                int wr = 0, hr = 0;
                binSorter->getItemRatio(it.w, it.h, wr, hr);
                std::string path = videoPool ? videoPool->getVideoPath(wr, hr) : "";
                VideoSlot slot;
                slot.x = (int)(baseX + it.x);
                slot.y = (int)(baseY + it.y);
                slot.w = it.w;
                slot.h = it.h;
                slot.ratioW = wr;
                slot.ratioH = hr;
                slot.path = path;
                if (!path.empty()) {
                    if (slot.player.load(path)) {
                        slot.player.setLoopState(OF_LOOP_NORMAL);
                        slot.player.play();
                        slot.hasVideo = true;
                    } else {
                        slot.hasVideo = false;
                        ofLogWarning("BinSorterRenderer") << "load failed for ratio "
                            << wr << "_" << hr << ": " << path;
                    }
                } else {
                    slot.hasVideo = false;
                }
                float slotAspect = (slot.h > 0) ? (float)slot.w / slot.h : 0;
                ofLogNotice("BinSorterRenderer") << "Slot " << (slots.size() + 1) << ": "
                    << slot.w << "x" << slot.h << " aspect=" << slotAspect
                    << " -> ratio " << wr << ":" << hr << (path.empty() ? "" : " ") << path;
                slots.push_back(slot);
            }
        }
    }
}

void BinSorterRenderer::update() {
    for (auto& slot : slots)
        if (slot.hasVideo)
            slot.player.update();
}

void BinSorterRenderer::draw(float offsetX, float offsetY) {
    if (!binSorter) return;
    int boxW = binSorter->getBoxWidth();
    int boxH = binSorter->getBoxHeight();
    auto& bins = binSorter->getBins();
    const int padding = 20;

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
