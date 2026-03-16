#pragma once

#include "BinSorter.h"
#include "ofVideoPlayer.h"
#include <vector>
#include <map>
#include <set>

struct VideoSlot {
    ofVideoPlayer player;      // currently displaying
    ofVideoPlayer nextPlayer;  // preloaded for seamless swap (only used when !videoLoop)
    std::string path;
    std::string nextPath;      // path of preloaded next video
    int x, y, w, h;
    int ratioW, ratioH;
    bool hasVideo = false;
};

class BinSorterRenderer {
public:
    void setup(BinSorter* sorter, class VideoAssetPool* pool, bool videoLoop = false);
    void update();
    void draw(float offsetX = 0, float offsetY = 0);
    void drawToFbo(ofFbo& fbo);
    void regenerate();
    void preloadFromArrangement(const Arrangement& arr);
    void swapToPreloaded(const Arrangement& arr);
    std::vector<VideoSlot>& getSlots() { return slots; }
    const std::vector<VideoSlot>& getSlots() const { return slots; }
    bool hasPreloadedLayout() const { return !nextSlots.empty(); }
private:
    void buildSlots();
    void buildSlotsFromArrangement(const std::vector<std::vector<BinItem>>& bins,
        const std::map<std::pair<int, int>, NestedBinData>& nestedBins,
        std::vector<VideoSlot>& out);

    BinSorter* binSorter = nullptr;
    VideoAssetPool* videoPool = nullptr;
    bool videoLoop = false;
    std::vector<VideoSlot> slots;
    std::vector<VideoSlot> nextSlots;
    std::set<std::string> loggedNotReadyKeys;
};
