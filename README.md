# Slate

An infinite canvas note-taking application for iOS built with Metal and SwiftUI, featuring smooth stroke rendering and high-performance pan/zoom/rotation navigation.

![iOS](https://img.shields.io/badge/iOS-15.0+-blue.svg)
![Swift](https://img.shields.io/badge/Swift-5.5+-orange.svg)
![Metal](https://img.shields.io/badge/Metal-3-red.svg)

## Overview

Slate demonstrates how to build a performant drawing app using Metal's GPU acceleration. The app achieves 60fps rendering with responsive 240Hz touch input and accurate coordinate handling at any zoom level.

## Features

- ✅ **Smooth Stroke Rendering** - Catmull-Rom spline interpolation for high-quality curves
- ✅ **Two-Finger Pan Navigation** - Intuitive canvas navigation without interfering with drawing
- ✅ **Pinch-to-Zoom and Rotation** - Smooth scaling with maintained stroke accuracy
- ✅ **GPU-Accelerated Transforms** - Pan and zoom operations handled entirely on GPU
- ✅ **Accurate Touch Positioning** - Precise coordinate space management across all zoom levels
- ✅ **High Performance** - 60fps rendering with 240Hz touch input capture

## Architecture Highlights

### Core Components

- **TouchableMTKView**: Custom `MTKView` subclass that captures raw touch events and handles gesture recognition
- **Coordinator Pattern**: Bridges SwiftUI and UIKit, managing Metal state and the render loop
- **GPU Transform System**: View transformations (pan/zoom) applied in vertex shader for optimal performance
- **Tessellation Engine**: Converts smooth Catmull-Rom curves into GPU-renderable triangle strips
- **Coordinate Space Management**: Precise conversions between screen space, world space, and normalized device coordinates

### Key Technical Decisions

**GPU-Side Transforms**: Strokes are stored in world coordinates and tessellated once. The current pan/zoom transform is passed to the GPU via a uniform buffer and applied in the vertex shader. This reduced frame time from ~37ms to <1ms when transforming multiple strokes.

**Touch Event Separation**: Raw touch capture happens in the `UIView` layer, while touch processing logic lives in the `Coordinator` with access to Metal state. This separation provides clean architecture and proper state management.

**Hybrid Stroke Rendering**: Round end caps with flat middle segments and round joints prevent visual gaps while maintaining rendering efficiency.

## Challenges & Learning

This project involved navigating largely undocumented territory at the intersection of Metal, SwiftUI, and touch handling. Key challenges included:

### Coordinate Space Management
Managing three coordinate systems (screen space with UIKit's Y-down origin, world space for stroke storage, and NDC with Y-up for GPU rendering) required careful mathematical transforms. The critical insight was ensuring the CPU's `screenToWorld()` inverse transform exactly mirrors the GPU's forward transform operations.

### Transform Order Dependencies
Getting pan and zoom to work correctly required understanding that transform order matters: zoom must be applied before pan in the vertex shader, and the inverse operations must happen in reverse order (unpan before unzoom) when converting touch coordinates.

### Performance Optimization
Initial attempts at CPU-side transform application re-tessellated all strokes every frame, creating a performance bottleneck. Moving transforms to the GPU was essential for maintaining 60fps with multiple strokes.

### Touch Event Handling
Capturing raw touch events at 240Hz while preventing gesture recognizers from interfering with single-finger drawing required careful configuration of gesture recognizer delegates and touch filtering logic.

### Lack of Documentation
There are very few resources covering the intersection of Metal rendering, SwiftUI integration, and complex touch handling. Most of the solutions required experimentation and deep understanding of coordinate systems. I learned heavily from the [30 Days of Metal](https://github.com/warren-bank/fork-30-days-of-metal) repository for Metal fundamentals, but much of the architecture had to be designed from first principles.

## Current State

The app is fully functional with smooth drawing, pan, and zoom working correctly at 60fps. Touch positioning is accurate at any zoom level, and the architecture is set up to handle additional features.

## Future Possibilities

Some potential directions for further development:
- Undo/redo functionality
- Stroke color and width selection
- Export to image or vector formats
- Stroke erasing
- Multi-touch drawing (multiple simultaneous strokes)
- Optimized rendering for very large numbers of strokes
- Pressure sensitivity support (Apple Pencil)

## Feedback & Contributions

**I'm actively seeking feedback and advice!** This project has been a deep dive into areas with limited documentation, and I'm sure there are better approaches to some of the challenges I've solved. If you have experience with:

- Metal rendering optimization
- SwiftUI and UIKit integration patterns
- Touch event handling and gesture recognition
- Coordinate space transforms
- Drawing app architecture

...I'd love to hear your thoughts. Open an issue or reach out if you have suggestions, questions, or want to discuss any aspect of the implementation.

## Getting Started

### Requirements
- iOS 15.0+
- Xcode 13.0+
- A device with Metal support (all iOS devices since 2013)

### Building
1. Clone the repository
2. Open `Slate.xcodeproj` in Xcode
3. Build and run on a physical device (Metal rendering performs best on hardware)

## Technical Details

### Coordinate Transform Pipeline

```
Touch Input (screen pixels, Y-down)
    ↓
screenToWorld() inverse transform
    ↓
World coordinates (stored in stroke)
    ↓
Tessellate to triangles (at identity transform)
    ↓
Stroke-local vertices (stored as offsets relative to stroke.origin)
    ↓
GPU vertex shader applies current pan/zoom
    ↓
Screen display (correct position at any transform)
```

### Performance Characteristics

- Touch input: 240Hz (raw UITouch events)
- Rendering: 60fps (Metal display refresh)
- Tessellation: One-time per completed stroke + per-frame for current in-progress stroke
- Transform application: GPU parallel processing


## Acknowledgments

- [30 Days of Metal](https://github.com/warren-bank/fork-30-days-of-metal) for Metal fundamentals
- The Metal and SwiftUI communities for scattered bits of wisdom

---

**Note**: This is a learning project exploring the boundaries of Metal and SwiftUI integration. The code prioritizes understanding over production polish, and I'm open to suggestions for improvements!
