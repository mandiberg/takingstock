#pragma once

#include <vector>
#include <map>
#include <utility>
#include <string>

struct SizeRatio {
    int w;
    int h;
    float weight;
    float expandX;
    float expandY;
    SizeRatio(int w_, int h_, float weight_ = 1.0f, float exX = 0.0f, float exY = 0.0f)
        : w(w_), h(h_), weight(weight_), expandX(exX), expandY(exY) {}
};

struct BinItem {
    int x, y, w, h;
    int itemIdx;
    BinItem(int x_ = 0, int y_ = 0, int w_ = 0, int h_ = 0, int idx = 0)
        : x(x_), y(y_), w(w_), h(h_), itemIdx(idx) {}
};

struct NestedBinData {
    std::vector<BinItem> items;
    int parentX, parentY, parentW, parentH;
    std::map<std::pair<int, int>, NestedBinData> nestedBins;
};

struct Arrangement {
    std::vector<std::vector<BinItem>> bins;
    std::map<std::pair<int, int>, NestedBinData> nestedBins;
};

class BinSorter {
public:
    BinSorter(int boxWidth, int boxHeight,
              const std::vector<SizeRatio>& sizeRatios,
              int packingStopArea = 1000,
              int nestingLayers = 0,
              int nestedMinSpaceThreshold = 100,
              float mainBinFillChance = 0.05f,
              float itemBreakScale = 0.0f,
              float itemBreakChance = 0.0f,
              int breakBoxMinItems = 2,
              int breakBoxMaxItems = 6,
              int breakBoxFillAttempts = 5,
              float breakBoxCoverageThreshold = 0.99f,
              float placementAreaExponent = 1.2f,
              int placementTopK = 1);

    void sort(int nestedStartItemIdx = -1);
    std::vector<std::vector<BinItem>>& getBins() { return bins; }
    const std::vector<std::vector<BinItem>>& getBins() const { return bins; }
    std::map<std::pair<int, int>, NestedBinData>& getNestedBins() { return nestedBins; }
    const std::map<std::pair<int, int>, NestedBinData>& getNestedBins() const { return nestedBins; }
    int getBoxWidth() const { return boxWidth; }
    int getBoxHeight() const { return boxHeight; }
    int getTotalItemsCount() const;
    int getTotalBinsCount() const;
    bool getItemRatio(int width, int height, int& outWr, int& outHr) const;
    std::string getLayoutSignature() const;
    /// Returns the area of the largest empty rectangle across all bins (for gap filtering)
    int getLargestFittableAreaInLayout() const;
    void loadArrangement(const std::vector<std::vector<BinItem>>& bins_,
                         const std::map<std::pair<int, int>, NestedBinData>& nestedBins_);

private:
    std::vector<SizeRatio> sizeRatios;
    int packingStopArea;
    int nestingLayers;
    int nestedMinSpaceThreshold;
    float mainBinFillChance;
    float itemBreakScale;
    float itemBreakChance;
    int breakBoxMinItems;
    int breakBoxMaxItems;
    int breakBoxFillAttempts;
    float breakBoxCoverageThreshold;
    float placementAreaExponent;
    int placementTopK;

    std::vector<std::vector<BinItem>> bins;
    std::map<std::pair<int, int>, NestedBinData> nestedBins;
    std::vector<float> cumulativeWeights;
public:
    std::pair<int, int> excludeRatioForFirstItem;
    bool hasExcludeRatio;
    int boxWidth, boxHeight;  // mutable for nested sorter override
private:

    void normalizeWeights();
    std::pair<int, int> selectWeightedRatio();
    std::pair<float, float> getExpandAllowances(int wr, int hr);
    std::tuple<int, int, int, int> gapsForItem(const std::vector<BinItem>& binItems,
                                                const BinItem& item, int boxW, int boxH, int tolerance = 1);
    std::vector<BinItem> gapFillPass(std::vector<BinItem> binItems, int boxW, int boxH);
    std::vector<BinItem> stretchToEdges(std::vector<BinItem> binItems, int boxW, int boxH);
    std::vector<SizeRatio> getRandomRestrictedRatios(int parentWr, int parentHr);
    float coverageFraction(const std::vector<std::tuple<int,int,int,int>>& placements, int boxW, int boxH);
    std::vector<std::tuple<int,int,int,int>> tryLayoutNItems(int boxW, int boxH, int n,
                                                             const std::vector<std::pair<int,int>>& ratiosOnly);
    std::vector<std::tuple<int,int,int,int>> fixOverlaps(const std::vector<std::tuple<int,int,int,int>>& placements,
                                                         int boxW, int boxH);
    bool canPlaceInBin(const std::vector<BinItem>& binItems, int itemW, int itemH, int binW, int binH) const;
    std::pair<int, int> findPosition(const std::vector<BinItem>& binItems, int itemW, int itemH, int binW, int binH) const;
    bool overlaps(int nx, int ny, int nw, int nh, const std::vector<BinItem>& existing) const;
    int getLargestFittableRectangle(const std::vector<BinItem>& binItems) const;
    int getLargestFittableRectangle(const std::vector<BinItem>& binItems, int boxW, int boxH) const;

    struct PlaceableResult {
        int x, y, w, h;
        int ratioW, ratioH;
        bool valid;
        PlaceableResult() : valid(false) {}
    };
    PlaceableResult findLargestPlaceableItem(const std::vector<BinItem>& binItems, bool excludeRatioForFirst) const;
    PlaceableResult findLargestPlaceableItem(const std::vector<BinItem>& binItems, bool excludeRatioForFirst, int boxW, int boxH) const;

    void createNestedBins(int remainingLayers, int startItemIdx);
    std::map<std::pair<int,int>, NestedBinData> remapNestedBinData(
        const std::map<std::pair<int,int>, NestedBinData>& nestedBinsDict,
        int offset, int binIdx, int parentItemIdx,
        const std::vector<BinItem>* remappedItemsList);
    int countNestedItems(const std::map<std::pair<int,int>, NestedBinData>& dict) const;
    int countNestedBins(const std::map<std::pair<int,int>, NestedBinData>& dict) const;
    int getMaxGapInNestedData(const NestedBinData& nd) const;
    /// True largest empty rectangle area (ignores aspect ratios)
    static int getLargestEmptyRectArea(const std::vector<BinItem>& items, int boxW, int boxH);

    std::vector<std::vector<BinItem>> sortInfinite();
};
