# Slate

An infinite canvas note-taking application for iOS built with Metal and SwiftUI, featuring smooth stroke rendering and high-performance pan/zoom/rotation navigation.

![iOS](https://img.shields.io/badge/iOS-15.0+-blue.svg)
![Swift](https://img.shields.io/badge/Swift-5.5+-orange.svg)
![Metal](https://img.shields.io/badge/Metal-3-red.svg)


## Overview

Slate demonstrates how to build a high-performance drawing app that breaks the traditional limits of floating-point arithmetic. By utilizing a custom "Telescoping" frame architecture, the app achieves 60fps rendering with 240Hz touch input at zoom levels exceeding $10^{1000}$ and beyond.

## Features

  - ✅ **True Infinite Zoom** - Zoom seamlessly from the macro to the micro (and back) without jitter or limits.
  - ✅ **Smooth Stroke Rendering** - Catmull-Rom spline interpolation for high-quality curves.
  - ✅ **Two-Finger Pan Navigation** - Intuitive canvas navigation without interfering with drawing.
  - ✅ **Recursive Rendering** - Visual continuity across depth levels (seeing parent drawings in the background).
  - ✅ **High Performance** - 60fps Metal rendering with precision-managed coordinate systems.

## Architecture Highlights

### Core Components

  - **Telescoping Coordinate System**: A recursive architecture where the "universe" resets every time you zoom in past a threshold. Instead of one global coordinate space, the app uses a linked list of nested `Frame` objects.
  - **Floating Origin & Local Realism**: Strokes are "locally aware" only. They store coordinates relative to their specific Frame, ensuring numbers stay small and precise regardless of the global zoom level.
  - **Recursive Compositor**: A rendering engine that traverses the Frame hierarchy, drawing the "Parent" frames (background) and "Child" frames (details) relative to the active camera view.
  - **TouchableMTKView**: Custom `MTKView` subclass that captures raw touch events and manages the complex "handoff" logic between coordinate systems.

### Key Technical Decisions

**Recursive Reference Frames (The "Engine"):**
Standard 64-bit `Double` precision fails around $10^{15}$ zoom (the "Double Precision Horizon"). To solve this, Slate uses a telescoping approach:

1.  When `zoomScale` exceeds 1,000x, the app "freezes" the current frame.
2.  It creates a new Child Frame centered on the user's gesture.
3.  It enters the new frame and resets `zoomScale` to 1.0.
    This ensures the math only ever processes values between 0.5 and 1,000, sidestepping overflow issues entirely.

**Local-Space Storage:**
Strokes are stored in **Local Coordinates** relative to their Frame's origin. They are ignorant of the "global" position. This "Local Realism" means a stroke at depth 50 ($10^{150}$ zoom) is mathematically identical to a stroke at depth 0, guaranteeing zero jitter.

**Anchor-Based Transitions:**
To make the "teleportation" between frames invisible to the user, the transition logic uses a shared **World Anchor** system. This locks the pixel under the user's finger to the exact same screen position during the coordinate swap, creating a seamless visual experience.

## Challenges & Learning

This project evolved from a standard infinite canvas into a research project on overcoming floating-point limitations.

### The "Large Coordinate" Problem

Initially, the app used a single global coordinate system. At zooms around $10^6$, strokes became jagged. At $10^{15}$, the camera shook violently due to precision loss (the "Event Horizon"). The solution was to abandon absolute space in favor of relative, nested spaces.


### Recursive Rendering Logic

Drawing a stroke that exists in a parent frame (1,000x larger) or a child frame (1,000x smaller) required deriving complex relative transforms. The renderer must "look up" to scale parent frames by multiplication, and "look down" to scale child frames by division, all while maintaining the illusion of a single continuous world.

## Technical Details

### The Telescoping Pipeline

```
Touch Input (Screen Pixels)
    ↓
[Coordinator] Check Zoom Threshold (> 1000x or < 0.5x)
    ↓
    IF Threshold Crossed:
    1. Create/Find Target Frame (Child or Parent)
    2. Calculate Finger Position in Target Frame
    3. RESET Zoom to 1.0 (relative to new frame)
    4. Solve Pan to lock Finger Position
    ↓
[Stroke Creation]
Convert Screen → Local Frame Coordinates (Double Precision)
    ↓
[Recursive Render Loop]
1. Render Parent (Background) → Scale UP (Zoom * 1000)
2. Render Active Frame (Foreground) → Scale Normal (Zoom * 1.0)
3. Render Children (Details) → Scale DOWN (Zoom / 1000)
    ↓
[Metal Shader]
Apply Relative Offset (Small Float) -> Final NDC Projection
```
## Getting Started

### Requirements

  - iOS 15.0+
  - Xcode 13.0+
  - A device with Metal support (all iOS devices since 2013)

### Building

1.  Clone the repository
2.  Open `Slate.xcodeproj` in Xcode
3.  Build and run on a physical device (Metal rendering performs best on hardware)

-----

**Note**: This project explores advanced coordinate system architectures. It prioritizes mathematical robustness and infinite scale over standard UI conventions.

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



## Acknowledgments

- [30 Days of Metal](https://github.com/warren-bank/fork-30-days-of-metal) for Metal fundamentals
- The Metal and SwiftUI communities for scattered bits of wisdom

---

**Note**: This is a learning project exploring the boundaries of Metal and SwiftUI integration. The code prioritizes understanding over production polish, and I'm open to suggestions for improvements!
