#pragma once

#include "ofMain.h"
#include "BinSorter.h"
#include <random>
#include "VideoAssetPool.h"
#include "BinSorterRenderer.h"
#include "ConfigLoader.h"
#include <memory>

class ofApp : public ofBaseApp {
public:
    void setup();
    void update();
    void draw();
    void keyPressed(int key);

private:
    BinSorterConfig config;
    std::unique_ptr<BinSorter> binSorter;
    VideoAssetPool videoPool;
    BinSorterRenderer renderer;
    ofFbo exportFbo;
    bool exportRequested = false;
    std::vector<Arrangement> arrangements;
    std::vector<size_t> pickQueue;
    bool arrangementPickRequested = false;
};
