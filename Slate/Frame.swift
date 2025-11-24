//
//  Frame.swift
//  Slate
//
//  Commit 1: The Data Model
//  Telescoping Reference Frames for Infinite Zoom
//
//  A "Frame" is a Local Universe - a coordinate system that stays bounded.
//  By chaining Frames together (parent → child), we achieve infinite zoom
//  without ever exceeding Double precision limits.
//

import Foundation

/// A Frame represents a bounded coordinate system (a "Local Universe").
///
/// **The Philosophy:**
/// Instead of one infinite coordinate system that breaks at 10^15, we use
/// a linked list of finite coordinate systems. When zoom exceeds a threshold,
/// we create a new Frame inside the current one and reset the zoom to 1.0.
///
/// **The Analogy:**
/// Like the movie "Men in Black" - the galaxy is inside a marble, which is
/// inside a bag, which is inside a locker. Each level is a separate Frame.
class Frame: Identifiable {
    let id = UUID()

    /// Strokes that belong to this specific "depth" or layer of reality
    var strokes: [Stroke] = []

    /// The "Universe" containing this frame (nil for root frame)
    var parent: Frame?

    /// Where this frame's center lives inside the Parent's coordinate system
    /// (Used to render this frame when the camera is in the Parent)
    var originInParent: SIMD2<Double>

    /// How much smaller this frame is compared to the Parent
    /// Example: If we drilled down at 1000× zoom, this would be 1000.0
    /// This tells us: "1 unit in parent space = 1000 units in this frame"
    var scaleRelativeToParent: Double

    /// 🟢 COMMIT 4 FIX: Track the sub-universes created inside this frame
    /// This allows us to re-enter existing frames instead of creating parallel universes
    var children: [Frame] = []

    /// Initialize a new Frame
    /// - Parameters:
    ///   - parent: The containing universe (nil for root)
    ///   - origin: Where this frame lives in parent coordinates
    ///   - scale: The zoom factor when this frame was created
    init(parent: Frame? = nil, origin: SIMD2<Double> = .zero, scale: Double = 1.0) {
        self.parent = parent
        self.originInParent = origin
        self.scaleRelativeToParent = scale
    }
}
