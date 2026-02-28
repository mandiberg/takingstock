#pragma once

#include "BinSorter.h"
#include "ofVideoPlayer.h"
#include <vector>
#include <map>
#include <set>

struct VideoSlot {
    ofVideoPlayer player;
    std::string path;
    int x, y, w, h;
    int ratioW, ratioH;
    bool hasVideo = false;
};

class BinSorterRenderer {
public:
    void setup(BinSorter* sorter, class VideoAssetPool* pool);
    void update();
    void draw(float offsetX = 0, float offsetY = 0);
    void drawToFbo(ofFbo& fbo);
    void regenerate();
    std::vector<VideoSlot>& getSlots() { return slots; }
    const std::vector<VideoSlot>& getSlots() const { return slots; }
private:
    void buildSlots();

    BinSorter* binSorter = nullptr;
    VideoAssetPool* videoPool = nullptr;
    std::vector<VideoSlot> slots;
    std::set<std::string> loggedNotReadyKeys;
};
