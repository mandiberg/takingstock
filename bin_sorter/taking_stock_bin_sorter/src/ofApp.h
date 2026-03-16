#pragma once

#include "ofMain.h"
#include "BinSorter.h"
#include <random>
#include "VideoAssetPool.h"
#include "BinSorterRenderer.h"
#include "ConfigLoader.h"
#include <memory>

enum class TransitionState { Idle, FadeDown, HoldBlack, FadeUp };

class ofApp : public ofBaseApp {
public:
    void setup();
    void update();
    void draw();
    void keyPressed(int key);

private:
    void pickAndLoadArrangement(size_t idx);
    void swapToPreloadedAndLog(size_t idx);
    void logArrangementInfo(size_t idx);
    void preloadNextLayout();
    float scheduleNextTransition();
    size_t pickNextArrangementIndex();

    BinSorterConfig config;
    std::unique_ptr<BinSorter> binSorter;
    VideoAssetPool videoPool;
    BinSorterRenderer renderer;
    ofFbo exportFbo;
    bool exportRequested = false;
    std::vector<Arrangement> arrangements;
    std::vector<size_t> pickQueue;
    bool arrangementPickRequested = false;

    TransitionState transitionState = TransitionState::Idle;
    float transitionStartTime = 0.f;
    float nextTransitionTime = 0.f;
    size_t nextLayoutIdx = 0;  // preloaded layout index
};
