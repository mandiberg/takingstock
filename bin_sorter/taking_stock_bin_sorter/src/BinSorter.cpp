#include "BinSorter.h"
#include "ofMain.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <set>

static bool chanceToDo(float chance) {
    return ofRandom(1.0f) < chance;
}

BinSorter::BinSorter(int boxWidth_, int boxHeight_,
                     const std::vector<SizeRatio>& sizeRatios_,
                     int packingStopArea_,
                     int nestingLayers_,
                     int nestedMinSpaceThreshold_,
                     float mainBinFillChance_,
                     float itemBreakScale_,
                     float itemBreakChance_,
                     int breakBoxMinItems_,
                     int breakBoxMaxItems_,
                     int breakBoxFillAttempts_,
                     float breakBoxCoverageThreshold_,
                     float placementAreaExponent_,
                     int placementTopK_)
    : sizeRatios(sizeRatios_),
      packingStopArea(packingStopArea_),
      nestingLayers(nestingLayers_),
      nestedMinSpaceThreshold(nestedMinSpaceThreshold_),
      mainBinFillChance(mainBinFillChance_),
      itemBreakScale(itemBreakScale_),
      itemBreakChance(itemBreakChance_),
      breakBoxMinItems(breakBoxMinItems_),
      breakBoxMaxItems(breakBoxMaxItems_),
      breakBoxFillAttempts(breakBoxFillAttempts_),
      breakBoxCoverageThreshold(breakBoxCoverageThreshold_),
      placementAreaExponent(placementAreaExponent_),
      placementTopK(placementTopK_),
      hasExcludeRatio(false),
      boxWidth(boxWidth_), boxHeight(boxHeight_)
{
    normalizeWeights();
}

void BinSorter::normalizeWeights() {
    cumulativeWeights.clear();
    if (sizeRatios.empty()) return;

    float total = 0;
    for (const auto& r : sizeRatios)
        total += r.weight;
    if (total <= 0)
        total = 1.0f / sizeRatios.size();

    float cum = 0;
    for (const auto& r : sizeRatios) {
        cum += r.weight / total;
        cumulativeWeights.push_back(cum);
    }
}

std::pair<int, int> BinSorter::selectWeightedRatio() {
    if (sizeRatios.empty()) return {0, 0};
    float r = ofRandom(1.0f);
    for (size_t i = 0; i < cumulativeWeights.size(); ++i) {
        if (r <= cumulativeWeights[i])
            return {sizeRatios[i].w, sizeRatios[i].h};
    }
    return {sizeRatios.back().w, sizeRatios.back().h};
}

std::pair<float, float> BinSorter::getExpandAllowances(int wr, int hr) {
    for (const auto& r : sizeRatios) {
        if (r.w == wr && r.h == hr)
            return {r.expandX, r.expandY};
    }
    return {0.0f, 0.0f};
}

std::tuple<int, int, int, int> BinSorter::gapsForItem(
    const std::vector<BinItem>& binItems, const BinItem& item,
    int boxW, int boxH, int tolerance)
{
    int left = item.x;
    int right = boxW - (item.x + item.w);
    int bottom = item.y;
    int top = boxH - (item.y + item.h);

    auto vertOverlap = [&item](int oy, int oh) {
        return !(item.y + item.h <= oy || oy + oh <= item.y);
    };
    auto horzOverlap = [&item](int ox, int ow) {
        return !(item.x + item.w <= ox || ox + ow <= item.x);
    };

    for (const auto& o : binItems) {
        if (o.itemIdx == item.itemIdx) continue;
        if (vertOverlap(o.y, o.h)) {
            if (o.x + o.w <= item.x)
                left = std::min(left, item.x - (o.x + o.w));
            if (o.x >= item.x + item.w)
                right = std::min(right, o.x - (item.x + item.w));
        }
        if (horzOverlap(o.x, o.w)) {
            if (o.y + o.h <= item.y)
                bottom = std::min(bottom, item.y - (o.y + o.h));
            if (o.y >= item.y + item.h)
                top = std::min(top, o.y - (item.y + item.h));
        }
    }

    auto clamp = [tolerance](int g) { return g <= tolerance ? 0 : g; };
    return {clamp(left), clamp(right), clamp(top), clamp(bottom)};
}

std::vector<BinItem> BinSorter::gapFillPass(std::vector<BinItem> binItems, int boxW, int boxH) {
    if (binItems.empty()) return binItems;

    std::vector<size_t> indices(binItems.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
    std::sort(indices.begin(), indices.end(), [&binItems](size_t a, size_t b) {
        if (binItems[a].y != binItems[b].y) return binItems[a].y < binItems[b].y;
        return binItems[a].x < binItems[b].x;
    });

    for (size_t i : indices) {
        auto& it = binItems[i];
        int baseW = it.w, baseH = it.h;
        auto [leftG, rightG, topG, bottomG] = gapsForItem(binItems, it, boxW, boxH);

        int wr = 0, hr = 0;
        if (!getItemRatio(it.w, it.h, wr, hr)) continue;
        auto [expandX, expandY] = getExpandAllowances(wr, hr);
        if (expandX <= 0 && expandY <= 0) continue;

        int dwLeft = std::min(leftG, (int)(baseW * expandX));
        int dwRight = std::min(rightG, (int)(baseW * expandX));
        int dhBottom = std::min(bottomG, (int)(baseH * expandY));
        int dhTop = std::min(topG, (int)(baseH * expandY));

        int newX = it.x - dwLeft;
        int newY = it.y - dhTop;
        int newW = it.w + dwLeft + dwRight;
        int newH = it.h + dhBottom + dhTop;

        if (newX < 0) { newW += newX; newX = 0; }
        if (newY < 0) { newH += newY; newY = 0; }
        if (newX + newW > boxW) newW = boxW - newX;
        if (newY + newH > boxH) newH = boxH - newY;
        if (newW >= 1 && newH >= 1)
            binItems[i] = BinItem(newX, newY, newW, newH, it.itemIdx);
    }
    return binItems;
}

std::vector<BinItem> BinSorter::stretchToEdges(std::vector<BinItem> binItems, int boxW, int boxH) {
    const int maxGap = 16;  // extend items to fill gaps up to this many pixels
    for (size_t i = 0; i < binItems.size(); ++i) {
        auto& it = binItems[i];
        int gapRight = boxW - (it.x + it.w);
        int gapBottom = boxH - (it.y + it.h);
        if (gapRight <= 0 && gapBottom <= 0) continue;

        int wr = 0, hr = 0;
        if (!getItemRatio(it.w, it.h, wr, hr)) continue;
        auto [expandX, expandY] = getExpandAllowances(wr, hr);

        if (gapRight > 0 && gapRight <= maxGap && expandX > 0) {
            bool blocked = false;
            for (size_t j = 0; j < binItems.size() && !blocked; ++j) {
                if (j == i) continue;
                const auto& o = binItems[j];
                if (o.x + o.w <= it.x + it.w || o.x >= boxW) continue;
                if (o.y + o.h <= it.y || o.y >= it.y + it.h) continue;
                blocked = true;
            }
            if (!blocked) {
                int dw = std::min(gapRight, (int)(it.w * expandX));
                if (dw > 0) it.w += dw;
            }
        }

        gapBottom = boxH - (it.y + it.h);
        if (gapBottom > 0 && gapBottom <= maxGap && expandY > 0) {
            bool blocked = false;
            for (size_t j = 0; j < binItems.size() && !blocked; ++j) {
                if (j == i) continue;
                const auto& o = binItems[j];
                if (o.y + o.h <= it.y + it.h || o.y >= boxH) continue;
                if (o.x + o.w <= it.x || o.x >= it.x + it.w) continue;
                blocked = true;
            }
            if (!blocked) {
                int dh = std::min(gapBottom, (int)(it.h * expandY));
                if (dh > 0) it.h += dh;
            }
        }
    }
    return binItems;
}

std::vector<SizeRatio> BinSorter::getRandomRestrictedRatios(int parentWr, int parentHr) {
    std::vector<SizeRatio> other;
    for (const auto& r : sizeRatios) {
        if (r.w != parentWr || r.h != parentHr)
            other.push_back(r);
    }
    if (other.empty()) other = sizeRatios;
    if ((int)other.size() <= 2) return other;
    int k = (int)ofRandom(2, std::min(4, (int)other.size()) + 1);
    std::shuffle(other.begin(), other.end(), std::mt19937(std::random_device{}()));
    if ((int)other.size() > k) other.erase(other.begin() + k, other.end());
    return other;
}

float BinSorter::coverageFraction(const std::vector<std::tuple<int,int,int,int>>& placements,
                                  int boxW, int boxH) {
    if (placements.empty() || boxW <= 0 || boxH <= 0) return 0.0f;
    int64_t covered = 0;
    for (const auto& p : placements)
        covered += std::get<2>(p) * std::get<3>(p);
    return std::min(1.0f, (float)covered / (boxW * boxH));
}

std::vector<std::tuple<int,int,int,int>> BinSorter::tryLayoutNItems(
    int boxW, int boxH, int n, const std::vector<std::pair<int,int>>& ratiosOnly)
{
    if (n < 2 || n > 6 || (int)ratiosOnly.size() < n) return {};

    std::vector<std::pair<int,int>> layouts;
    if (n == 2) layouts = {{1,2}, {2,1}};
    else if (n == 3) layouts = {{1,3}, {3,1}};
    else if (n == 4) layouts = {{2,2}};
    else if (n == 5) layouts = {{1,5}, {5,1}};
    else if (n == 6) layouts = {{2,3}, {3,2}};
    else return {};

    for (auto [rows, cols] : layouts) {
        for (int attempt = 0; attempt < 50; ++attempt) {
            std::vector<std::pair<int,int>> selected;
            if ((int)ratiosOnly.size() >= n) {
                std::vector<size_t> idx(ratiosOnly.size());
                for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
                std::shuffle(idx.begin(), idx.end(), std::mt19937(std::random_device{}()));
                for (int i = 0; i < n; ++i) selected.push_back(ratiosOnly[idx[i]]);
            } else {
                for (int i = 0; i < n; ++i)
                    selected.push_back(ratiosOnly[(size_t)ofRandom(ratiosOnly.size())]);
            }

            std::vector<std::vector<std::pair<int,int>>> ratioGrid;
            for (int r = 0; r < rows; ++r) {
                ratioGrid.push_back({});
                for (int c = 0; c < cols; ++c)
                    ratioGrid.back().push_back(selected[r * cols + c]);
            }

            std::vector<float> rowHeights;
            for (const auto& rowRatios : ratioGrid) {
                float rowWidthRate = 0;
                for (auto [wr, hr] : rowRatios) {
                    if (hr > 0) rowWidthRate += (float)wr / hr;
                }
                if (rowWidthRate <= 0) break;
                rowHeights.push_back((float)boxW / rowWidthRate);
            }
            if ((int)rowHeights.size() != rows) continue;

            float totalH = 0;
            for (float h : rowHeights) totalH += h;
            if (totalH <= 0) continue;
            float scaleH = (float)boxH / totalH;
            for (float& h : rowHeights) h *= scaleH;

            std::vector<int> rowHeightsInt(rows);
            for (int i = 0; i < rows; ++i)
                rowHeightsInt[i] = std::max(1, (int)std::round(rowHeights[i]));
            int sumH = 0;
            for (int h : rowHeightsInt) sumH += h;
            if (sumH != boxH)
                rowHeightsInt.back() = std::max(1, boxH - sumH + rowHeightsInt.back());

            std::vector<std::tuple<int,int,int,int>> placements;
            int y = 0;
            bool valid = true;

            for (int rowIdx = 0; rowIdx < rows && valid; ++rowIdx) {
                const auto& rowRatios = ratioGrid[rowIdx];
                int rowH = rowHeightsInt[rowIdx];

                std::vector<int> itemWidths;
                for (auto [wr, hr] : rowRatios) {
                    if (hr <= 0) { valid = false; break; }
                    itemWidths.push_back(std::max(1, (int)std::round(rowH * wr / hr)));
                }
                if (!valid) break;

                int rowSum = 0;
                for (int w : itemWidths) rowSum += w;
                if (rowSum > 0 && rowSum != boxW) {
                    float scaleW = (float)boxW / rowSum;
                    for (int& w : itemWidths) w = std::max(1, (int)std::round(w * scaleW));
                    rowSum = 0;
                    for (int w : itemWidths) rowSum += w;
                    if (rowSum > 0 && rowSum != boxW) {
                        float scaleW2 = (float)boxW / rowSum;
                        for (int& w : itemWidths) w = std::max(1, (int)std::round(w * scaleW2));
                        rowSum = 0;
                        for (int w : itemWidths) rowSum += w;
                        if (std::abs(rowSum - boxW) <= 2)
                            itemWidths.back() = std::max(1, boxW - rowSum + itemWidths.back());
                    }
                }

                int x = 0;
                for (size_t i = 0; i < rowRatios.size(); ++i) {
                    int w = itemWidths[i];
                    int h = rowH;
                    if (x + w > boxW || y + h > boxH) { valid = false; break; }
                    placements.push_back({x, y, w, h});
                    x += w;
                }
                y += rowH;
            }

            if (valid && (int)placements.size() == n) {
                for (size_t i = 0; i < placements.size() && valid; ++i) {
                    auto [px, py, pw, ph] = placements[i];
                    auto [wr, hr] = selected[i];
                    if (hr == 0 || px + pw > boxW || py + ph > boxH) { valid = false; break; }
                    float aspectCurr = (float)pw / ph;
                    float aspectTarget = (float)wr / hr;
                    if (std::abs(aspectCurr - aspectTarget) / aspectTarget > 0.02f)
                        valid = false;
                }
                if (valid) return placements;
            }
        }
    }
    return {};
}

std::vector<std::tuple<int,int,int,int>> BinSorter::fixOverlaps(
    const std::vector<std::tuple<int,int,int,int>>& placements, int boxW, int boxH)
{
    if (placements.empty()) return placements;

    std::vector<std::tuple<int,int,int,int>> sorted = placements;
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        if (std::get<1>(a) != std::get<1>(b)) return std::get<1>(a) < std::get<1>(b);
        return std::get<0>(a) < std::get<0>(b);
    });

    std::vector<std::tuple<int,int,int,int>> fixed;
    for (auto [x, y, w, h] : sorted) {
        int maxYEnd = 0;
        for (const auto& [fx, fy, fw, fh] : fixed) {
            if (!(x + w <= fx || fx + fw <= x))
                maxYEnd = std::max(maxYEnd, fy + fh);
        }
        if (y < maxYEnd) y = maxYEnd;
        if (y + h > boxH) {
            h = std::max(1, boxH - y);
            if (h <= 0) continue;
        }
        fixed.push_back({x, y, w, h});
    }
    return fixed;
}

bool BinSorter::canPlaceInBin(const std::vector<BinItem>& binItems, int itemW, int itemH,
                              int binW, int binH) const {
    int64_t used = 0;
    for (const auto& it : binItems) used += it.w * it.h;
    return (int64_t)binW * binH - used >= (int64_t)itemW * itemH;
}

bool BinSorter::overlaps(int nx, int ny, int nw, int nh, const std::vector<BinItem>& existing) const {
    for (const auto& e : existing) {
        if (!(nx + nw <= e.x || e.x + e.w <= nx || ny + nh <= e.y || e.y + e.h <= ny))
            return true;
    }
    return false;
}

std::pair<int, int> BinSorter::findPosition(const std::vector<BinItem>& binItems,
                                            int itemW, int itemH, int binW, int binH) const {
    if (binItems.empty()) {
        std::vector<std::pair<int,int>> corners = {
            {0, 0},
            {binW - itemW, 0},
            {0, binH - itemH},
            {binW - itemW, binH - itemH}
        };
        std::vector<std::pair<int,int>> valid;
        for (auto [cx, cy] : corners) {
            if (cx >= 0 && cx + itemW <= binW && cy >= 0 && cy + itemH <= binH)
                valid.push_back({cx, cy});
        }
        if (!valid.empty())
            return valid[(size_t)ofRandom(valid.size())];
        return {0, 0};
    }

    std::vector<std::tuple<int,int,int>> candidates;

    int minX = binW;
    for (const auto& it : binItems) minX = std::min(minX, it.x);
    if (minX > 0) {
        for (int testY = 0; testY <= binH - itemH; testY += std::max(1, itemH / 4)) {
            if (!overlaps(0, testY, itemW, itemH, binItems) &&
                0 + itemW <= binW && testY + itemH <= binH)
                candidates.push_back({0, testY, -testY * 1000});
        }
    }

    std::vector<BinItem> sortedItems = binItems;
    std::sort(sortedItems.begin(), sortedItems.end(), [](const BinItem& a, const BinItem& b) {
        if (a.y != b.y) return a.y < b.y;
        return a.x < b.x;
    });

    for (const auto& it : sortedItems) {
        int newX = it.x + it.w;
        if (newX + itemW <= binW && it.y + itemH <= binH &&
            !overlaps(newX, it.y, itemW, itemH, binItems))
            candidates.push_back({newX, it.y, -newX * 100 - it.y});

        int newY = it.y + it.h;
        if (newY + itemH <= binH && it.x + itemW <= binW &&
            !overlaps(it.x, newY, itemW, itemH, binItems))
            candidates.push_back({it.x, newY, -it.x * 100 - newY});
    }

    int maxY = 0;
    for (const auto& it : binItems) maxY = std::max(maxY, it.y + it.h);
    if (maxY + itemH <= binH && !overlaps(0, maxY, itemW, itemH, binItems))
        candidates.push_back({0, maxY, -maxY});

    for (int testY = 0; testY <= binH - itemH; testY += std::max(1, itemH / 4)) {
        for (int testX = 0; testX <= binW - itemW; testX += std::max(1, itemW / 4)) {
            if (!overlaps(testX, testY, itemW, itemH, binItems) &&
                testX + itemW <= binW && testY + itemH <= binH)
                candidates.push_back({testX, testY, -testX * 100 - testY});
        }
    }

    if (!candidates.empty()) {
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& a, const auto& b) { return std::get<2>(a) > std::get<2>(b); });
        return {std::get<0>(candidates[0]), std::get<1>(candidates[0])};
    }
    return {-1, -1};
}

bool BinSorter::getItemRatio(int width, int height, int& outWr, int& outHr) const {
    if (sizeRatios.empty() || height <= 0) return false;
    float aspect = (float)width / height;
    float bestDiff = 1e9f;
    for (const auto& r : sizeRatios) {
        if (r.h <= 0) continue;
        float ratioAspect = (float)r.w / r.h;
        float diff = std::abs(aspect - ratioAspect);
        if (diff < bestDiff) {
            bestDiff = diff;
            outWr = r.w;
            outHr = r.h;
        }
    }
    return true;
}

BinSorter::PlaceableResult BinSorter::findLargestPlaceableItem(
    const std::vector<BinItem>& binItems, bool excludeRatioForFirst) const
{
    return findLargestPlaceableItem(binItems, excludeRatioForFirst, boxWidth, boxHeight);
}

BinSorter::PlaceableResult BinSorter::findLargestPlaceableItem(
    const std::vector<BinItem>& binItems, bool excludeRatioForFirst, int boxW, int boxH) const
{
    PlaceableResult result;
    if (sizeRatios.empty()) return result;

    std::pair<int,int> excludeRatio = {-1, -1};
    if (excludeRatioForFirst && hasExcludeRatio)
        excludeRatio = excludeRatioForFirstItem;

    float shrinkFactors[] = {1.0f, 0.98f, 0.95f, 0.92f, 0.9f, 0.87f, 0.85f, 0.82f, 0.8f, 0.77f, 0.75f,
                            0.72f, 0.7f, 0.67f, 0.65f, 0.62f, 0.6f, 0.55f, 0.5f, 0.45f, 0.4f, 0.35f,
                            0.3f, 0.25f, 0.2f, 0.15f, 0.1f};

    struct Candidate { int x, y, w, h, rw, rh; float score, area, weight; };
    std::vector<Candidate> candidates;

    std::vector<std::tuple<int,int,float>> weightedRatios;
    for (const auto& r : sizeRatios)
        weightedRatios.push_back({r.w, r.h, r.weight});
    std::sort(weightedRatios.begin(), weightedRatios.end(),
              [](const auto& a, const auto& b) { return std::get<2>(a) > std::get<2>(b); });

    for (const auto& [widthRatio, heightRatio, weight] : weightedRatios) {
        if (excludeRatio.first >= 0) {
            if (boxW == excludeRatio.first && boxH == excludeRatio.second) {
                if (widthRatio * excludeRatio.second == heightRatio * excludeRatio.first)
                    continue;
            } else if (widthRatio == excludeRatio.first && heightRatio == excludeRatio.second) {
                continue;
            }
        }

        int wRatio = widthRatio, hRatio = heightRatio;
        if (wRatio <= 0 || hRatio <= 0) continue;

        float maxScale = std::min((float)boxW / wRatio, (float)boxH / hRatio);
        if (maxScale <= 0) continue;

        for (float f : shrinkFactors) {
            int w = (int)(wRatio * maxScale * f);
            int h = (int)(hRatio * maxScale * f);
            if (w < 1 || h < 1) continue;

            if (binItems.size() == 1) {
                int binSmallest = std::min(boxW, boxH);
                if (std::max(w, h) > binSmallest) continue;
            }

            if (!canPlaceInBin(binItems, w, h, boxW, boxH)) continue;
            auto [x, y] = findPosition(binItems, w, h, boxW, boxH);
            if (x < 0) continue;

            float area = (float)(w * h);
            float score = std::pow(area, placementAreaExponent) * weight;
            candidates.push_back({x, y, w, h, wRatio, hRatio, score, area, weight});
            break;
        }
    }

    if (!candidates.empty()) {
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) {
                      if (a.score != b.score) return a.score > b.score;
                      return a.area > b.area;
                  });
        size_t pickIdx = 0;
        if (placementTopK > 1 && candidates.size() > 1) {
            int k = std::min(placementTopK, (int)candidates.size());
            float totalScore = 0;
            for (int i = 0; i < k; ++i) totalScore += candidates[i].score;
            if (totalScore > 0) {
                float r = ofRandom(totalScore);
                for (int i = 0; i < k; ++i) {
                    r -= candidates[i].score;
                    if (r <= 0) { pickIdx = i; break; }
                    if (i == k - 1) pickIdx = k - 1;
                }
            }
        }
        const auto& c = candidates[pickIdx];
        result.x = c.x;
        result.y = c.y;
        result.w = c.w;
        result.h = c.h;
        result.ratioW = c.rw;
        result.ratioH = c.rh;
        result.valid = true;
    }
    return result;
}

int BinSorter::getLargestFittableRectangle(const std::vector<BinItem>& binItems) const {
    return getLargestFittableRectangle(binItems, boxWidth, boxHeight);
}

int BinSorter::getLargestFittableRectangle(const std::vector<BinItem>& binItems, int boxW, int boxH) const {
    auto r = findLargestPlaceableItem(binItems, false, boxW, boxH);
    return r.valid ? r.w * r.h : 0;
}

int BinSorter::getLargestEmptyRectArea(const std::vector<BinItem>& items, int boxW, int boxH) {
    std::set<int> xs = {0, boxW};
    std::set<int> ys = {0, boxH};
    for (const auto& it : items) {
        xs.insert(it.x);
        xs.insert(it.x + it.w);
        ys.insert(it.y);
        ys.insert(it.y + it.h);
    }
    std::vector<int> xv(xs.begin(), xs.end());
    std::vector<int> yv(ys.begin(), ys.end());
    int maxArea = 0;
    for (size_t i = 0; i + 1 < xv.size(); ++i) {
        for (size_t j = 0; j + 1 < yv.size(); ++j) {
            int cx = xv[i], cy = yv[j];
            int cw = xv[i + 1] - cx, ch = yv[j + 1] - cy;
            if (cw <= 0 || ch <= 0) continue;
            bool empty = true;
            for (const auto& it : items) {
                if (!(cx + cw <= it.x || it.x + it.w <= cx || cy + ch <= it.y || it.y + it.h <= cy)) {
                    empty = false;
                    break;
                }
            }
            if (empty) {
                int area = cw * ch;
                if (area > maxArea) maxArea = area;
            }
        }
    }
    return maxArea;
}

int BinSorter::getMaxGapInNestedData(const NestedBinData& nd) const {
    int maxArea = getLargestEmptyRectArea(nd.items, nd.parentW, nd.parentH);
    for (const auto& kv : nd.nestedBins) {
        int deep = getMaxGapInNestedData(kv.second);
        if (deep > maxArea) maxArea = deep;
    }
    return maxArea;
}

int BinSorter::getLargestFittableAreaInLayout() const {
    int maxArea = 0;
    for (size_t bi = 0; bi < bins.size(); ++bi) {
        int g = getLargestEmptyRectArea(bins[bi], boxWidth, boxHeight);
        if (g > maxArea) maxArea = g;
        for (const auto& it : bins[bi]) {
            auto nit = nestedBins.find({(int)bi, it.itemIdx});
            if (nit != nestedBins.end()) {
                int deep = getMaxGapInNestedData(nit->second);
                if (deep > maxArea) maxArea = deep;
            }
        }
    }
    return maxArea;
}

std::string BinSorter::getLayoutSignature() const {
    std::string sig;
    for (size_t bi = 0; bi < bins.size(); ++bi) {
        if (bi > 0) sig += "|";
        for (size_t ii = 0; ii < bins[bi].size(); ++ii) {
            if (ii > 0) sig += ";";
            const auto& it = bins[bi][ii];
            sig += std::to_string(it.w) + "x" + std::to_string(it.h);
            auto nit = nestedBins.find({(int)bi, it.itemIdx});
            if (nit != nestedBins.end()) {
                sig += "[";
                for (size_t ni = 0; ni < nit->second.items.size(); ++ni) {
                    if (ni > 0) sig += ",";
                    const auto& n = nit->second.items[ni];
                    sig += std::to_string(n.w) + "x" + std::to_string(n.h);
                }
                sig += "]";
            }
        }
    }
    return sig;
}

int BinSorter::countNestedItems(const std::map<std::pair<int,int>, NestedBinData>& dict) const {
    int total = 0;
    for (const auto& kv : dict) {
        total += kv.second.items.size();
        total += countNestedItems(kv.second.nestedBins);
    }
    return total;
}

int BinSorter::countNestedBins(const std::map<std::pair<int,int>, NestedBinData>& dict) const {
    int total = 0;
    for (const auto& kv : dict) {
        total += 1;
        total += countNestedBins(kv.second.nestedBins);
    }
    return total;
}

std::vector<std::vector<BinItem>> BinSorter::sortInfinite() {
    std::vector<std::vector<BinItem>> result = {{}};
    if (sizeRatios.empty()) return result;

    int itemIdx = 0;
    bool isFirstItem = true;
    bool alreadyBroke = false;  // At most one breakbox per layout

    while (true) {
        int largestArea = getLargestFittableRectangle(result.back());
        if (packingStopArea > 0 && largestArea < packingStopArea) break;

        if (isFirstItem && !hasExcludeRatio) {
            if (chanceToDo(mainBinFillChance))
                hasExcludeRatio = false;
            else {
                excludeRatioForFirstItem = {boxWidth, boxHeight};
                hasExcludeRatio = true;
            }
        }

        auto best = findLargestPlaceableItem(result.back(), isFirstItem);
        if (!best.valid) break;

        int x = best.x, y = best.y, w = best.w, h = best.h;
        int wr = best.ratioW, hr = best.ratioH;
        int area = w * h;
        int binArea = boxWidth * boxHeight;
        float breakThreshold = itemBreakScale * binArea;

        bool doBreak = (!alreadyBroke && itemBreakScale > 0 && area >= breakThreshold && chanceToDo(itemBreakChance));

        if (doBreak) {
            result.back().push_back(BinItem(x, y, w, h, itemIdx));
            auto restricted = getRandomRestrictedRatios(wr, hr);
            std::vector<std::pair<int,int>> ratiosOnly;
            for (const auto& r : restricted) ratiosOnly.push_back({r.w, r.h});

            int minN = std::max(2, std::min(breakBoxMinItems, breakBoxMaxItems));
            int maxN = std::max(minN, breakBoxMaxItems);
            int target = (int)ofRandom(minN, maxN + 1);

            std::vector<std::tuple<int,int,int,int>> placements;
            for (int n = target; n >= minN; --n) {
                for (int attempt = 0; attempt < breakBoxFillAttempts; ++attempt) {
                    placements = tryLayoutNItems(w, h, n, ratiosOnly);
                    if (placements.empty()) {
                        BinSorter nested(w, h, restricted, 0, 0, nestedMinSpaceThreshold,
                                         0.05f, 0, 0, breakBoxMinItems, breakBoxMaxItems,
                                         breakBoxFillAttempts, breakBoxCoverageThreshold,
                                         placementAreaExponent, placementTopK);
                        nested.excludeRatioForFirstItem = {wr, hr};
                        nested.hasExcludeRatio = true;
                        nested.sort(-1);
                        if (!nested.bins.empty() && (int)nested.bins[0].size() >= n) {
                            for (int i = 0; i < n; ++i) {
                                const auto& it = nested.bins[0][i];
                                placements.push_back({it.x, it.y, it.w, it.h});
                            }
                        }
                    }
                    if (!placements.empty()) {
                        if (coverageFraction(placements, w, h) < breakBoxCoverageThreshold) {
                            placements.clear();
                            continue;
                        }
                        placements = fixOverlaps(placements, w, h);
                        break;
                    }
                }
                if (!placements.empty()) break;
            }

            if (!placements.empty()) {
                std::vector<BinItem> remapped;
                for (size_t i = 0; i < placements.size(); ++i) {
                    auto [nx, ny, nw, nh] = placements[i];
                    remapped.push_back(BinItem(nx, ny, nw, nh, itemIdx + 1 + (int)i));
                }
                NestedBinData nd;
                nd.items = remapped;
                nd.parentX = x; nd.parentY = y; nd.parentW = w; nd.parentH = h;
                nestedBins[{0, itemIdx}] = nd;
                itemIdx += 1 + (int)placements.size();
                alreadyBroke = true;
            } else {
                result.back().pop_back();
                result.back().push_back(BinItem(x, y, w, h, itemIdx));
                itemIdx++;
            }
        } else {
            result.back().push_back(BinItem(x, y, w, h, itemIdx));
            itemIdx++;
        }
        isFirstItem = false;
    }
    return result;
}

void BinSorter::sort(int nestedStartItemIdx) {
    bins = sortInfinite();

    if (nestingLayers > 0)
        createNestedBins(nestingLayers, nestedStartItemIdx);

    for (size_t i = 0; i < bins.size(); ++i) {
        bins[i] = gapFillPass(bins[i], boxWidth, boxHeight);
        bins[i] = stretchToEdges(bins[i], boxWidth, boxHeight);
    }

    for (auto& kv : nestedBins) {
        int binIdx = kv.first.first;
        int parentIdx = kv.first.second;
        int pw = kv.second.parentW, ph = kv.second.parentH;
        // Use parent's current dimensions after gapFillPass (they may have been expanded)
        if (binIdx >= 0 && binIdx < (int)bins.size()) {
            for (const auto& pit : bins[binIdx]) {
                if (pit.itemIdx == parentIdx) {
                    pw = pit.w;
                    ph = pit.h;
                    break;
                }
            }
        }
        auto items = gapFillPass(kv.second.items, pw, ph);
        items = stretchToEdges(items, pw, ph);
        std::vector<std::tuple<int,int,int,int>> placements;
        for (const auto& it : items) placements.push_back({it.x, it.y, it.w, it.h});
        auto fixed = fixOverlaps(placements, pw, ph);
        kv.second.items.clear();
        for (size_t i = 0; i < fixed.size(); ++i) {
            auto [fx, fy, fw, fh] = fixed[i];
            kv.second.items.push_back(BinItem(fx, fy, fw, fh, items[i].itemIdx));
        }
    }
}

void BinSorter::createNestedBins(int remainingLayers, int startItemIdx) {
    if (remainingLayers <= 0) return;

    if (startItemIdx < 0) {
        startItemIdx = 0;
        for (const auto& bin : bins) startItemIdx += bin.size();
    }

    for (size_t binIdx = 0; binIdx < bins.size(); ++binIdx) {
        auto& binItems = bins[binIdx];
        if (binItems.empty()) continue;

        std::vector<std::pair<BinItem, int>> itemsWithArea;
        for (const auto& it : binItems)
            itemsWithArea.push_back({it, it.w * it.h});
        std::sort(itemsWithArea.begin(), itemsWithArea.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        std::vector<std::pair<BinItem, int>> candidates;
        for (size_t i = 0; i < std::min((size_t)2, itemsWithArea.size()); ++i) {
            const auto& [item, area] = itemsWithArea[i];
            int w = item.w, h = item.h;
            bool canFitMultiple = false;
            for (const auto& r : sizeRatios) {
                float scaleW = w / (float)r.w;
                float scaleH = h / (float)r.h;
                float scale = std::min(scaleW, scaleH);
                if (scale <= 0) continue;
                int iw = (int)(r.w * scale), ih = (int)(r.h * scale);
                if (iw < 1 || ih < 1) continue;
                if ((iw * 2 <= w && ih <= h) || (ih * 2 <= h && iw <= w)) {
                    canFitMultiple = true;
                    break;
                }
            }
            if (canFitMultiple) candidates.push_back(itemsWithArea[i]);
        }
        if (candidates.empty())
            candidates = std::vector<std::pair<BinItem,int>>(itemsWithArea.begin(),
                                                             itemsWithArea.begin() + std::min((size_t)2, itemsWithArea.size()));

        std::shuffle(candidates.begin(), candidates.end(), std::mt19937(std::random_device{}()));

        for (const auto& [candidateItem, _] : candidates) {
            int x = candidateItem.x, y = candidateItem.y, w = candidateItem.w, h = candidateItem.h;
            int itemIdx = candidateItem.itemIdx;

            int parentWr = 0, parentHr = 0;
            if (!getItemRatio(w, h, parentWr, parentHr)) continue;

            bool nestedSuccess = false;
            int numNested = 0;
            std::vector<BinItem> remappedItems;
            std::map<std::pair<int,int>, NestedBinData> translatedNested;

            float scaleFactors[] = {1.0f, 0.8f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f};
            for (int excludeParent = 1; excludeParent >= 0 && !nestedSuccess; --excludeParent) {
                for (float sf : scaleFactors) {
                    int scaledThreshold = (int)(nestedMinSpaceThreshold * sf);
                    BinSorter nested(w, h, sizeRatios, scaledThreshold, remainingLayers - 1,
                                    nestedMinSpaceThreshold, 0.05f, 0, 0,
                                    breakBoxMinItems, breakBoxMaxItems, breakBoxFillAttempts,
                                    breakBoxCoverageThreshold,
                                    placementAreaExponent, placementTopK);
                    if (excludeParent) {
                        nested.excludeRatioForFirstItem = {parentWr, parentHr};
                        nested.hasExcludeRatio = true;
                    }
                    nested.sort(startItemIdx);
                    numNested = nested.bins.empty() ? 0 : nested.bins[0].size();
                    if (numNested >= 2) {
                        nestedSuccess = true;
                        for (const auto& it : nested.bins[0])
                            remappedItems.push_back(BinItem(it.x, it.y, it.w, it.h, startItemIdx + it.itemIdx));
                        translatedNested = remapNestedBinData(nested.nestedBins, startItemIdx,
                                                              (int)binIdx, itemIdx, &remappedItems);
                        break;
                    }
                }
            }

            if (!nestedSuccess) continue;

            NestedBinData nd;
            nd.items = remappedItems;
            nd.parentX = x; nd.parentY = y; nd.parentW = w; nd.parentH = h;
            nd.nestedBins = translatedNested;
            nestedBins[{(int)binIdx, itemIdx}] = nd;
            break;
        }
    }
}

std::map<std::pair<int,int>, NestedBinData> BinSorter::remapNestedBinData(
    const std::map<std::pair<int,int>, NestedBinData>& nestedBinsDict,
    int offset, int binIdx, int parentItemIdx, const std::vector<BinItem>* remappedItemsList)
{
    std::map<std::pair<int,int>, NestedBinData> remapped;
    for (const auto& kv : nestedBinsDict) {
        int oldItemIdx = kv.first.second;
        int remappedItemIdx;
        if (remappedItemsList && !remappedItemsList->empty()) {
            if (oldItemIdx >= 0 && oldItemIdx < (int)remappedItemsList->size())
                remappedItemIdx = (*remappedItemsList)[oldItemIdx].itemIdx;
            else
                remappedItemIdx = offset + oldItemIdx;
        } else {
            remappedItemIdx = offset + oldItemIdx;
        }

        std::pair<int,int> key = {remappedItemIdx, 0};
        NestedBinData nd = kv.second;
        nd.items.clear();
        for (const auto& it : kv.second.items)
            nd.items.push_back(BinItem(it.x, it.y, it.w, it.h, offset + it.itemIdx));
        nd.nestedBins = remapNestedBinData(kv.second.nestedBins, offset, binIdx, remappedItemIdx, &nd.items);
        remapped[key] = nd;
    }
    return remapped;
}

int BinSorter::getTotalItemsCount() const {
    int main = 0;
    for (const auto& bin : bins) main += bin.size();
    return main + countNestedItems(nestedBins);
}

int BinSorter::getTotalBinsCount() const {
    return bins.size() + countNestedBins(nestedBins);
}

void BinSorter::loadArrangement(const std::vector<std::vector<BinItem>>& bins_,
                                const std::map<std::pair<int, int>, NestedBinData>& nestedBins_) {
    bins = bins_;
    nestedBins = nestedBins_;
}
