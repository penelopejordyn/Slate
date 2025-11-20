//
//  Stroke.swift
//  Slate
//
//  Created by Penny Marshall on 10/28/25.
//

import SwiftUI

/// A stroke on the infinite canvas using Floating Origin architecture.
///
/// **Key Concept:** Instead of storing absolute world coordinates (which cause precision
/// issues at high zoom), we store:
/// - An `origin` (anchor point) in Double precision (absolute world coords)
/// - All vertices as Float offsets from that origin (local coords)
///
/// This ensures the GPU only ever receives small Float values, eliminating precision gaps.
struct Stroke {
    let id: UUID
    let origin: SIMD2<Double>           // Anchor point in world space (Double precision)
    let localVertices: [SIMD2<Float>]   // Vertices relative to origin (Float precision)
    let worldWidth: Double              // Width in world units
    let color: SIMD4<Float>

    /// Initialize stroke from screen-space points using direct delta calculation.
    /// This avoids Double precision loss at extreme zoom levels by calculating
    /// the stroke geometry directly from screen-space deltas rather than converting
    /// all points to absolute world coordinates.
    ///
    /// - Parameters:
    ///   - screenPoints: Raw screen points (high precision)
    ///   - zoomAtCreation: Zoom level when stroke was drawn
    ///   - panAtCreation: Pan offset when stroke was drawn
    ///   - viewSize: View dimensions
    ///   - rotationAngle: Camera rotation angle
    ///   - color: Stroke color
    /// 🟢 UPGRADED: Now accepts Double for zoom and pan to maintain precision
    init(screenPoints: [CGPoint],
         zoomAtCreation: Double,
         panAtCreation: SIMD2<Double>,
         viewSize: CGSize,
         rotationAngle: Float,
         color: SIMD4<Float>) {
        self.id = UUID()
        self.color = color

        guard let firstScreenPt = screenPoints.first else {
            self.origin = .zero
            self.localVertices = []
            self.worldWidth = 0
            return
        }

        // 1. CALCULATE ORIGIN (ABSOLUTE) - 🟢 HIGH PRECISION FIX
        // We still need the absolute world position for the anchor, so we know WHERE the stroke is.
        // Only convert the FIRST point to world coordinates.
        // Use the Pure Double helper so the anchor is precise at 1,000,000x zoom
        self.origin = screenToWorldPixels_PureDouble(firstScreenPt,
                                                     viewSize: viewSize,
                                                     panOffset: panAtCreation,
                                                     zoomScale: zoomAtCreation,
                                                     rotationAngle: rotationAngle)

        // 2. CALCULATE GEOMETRY (RELATIVE) - THE FIX for Double precision
        // 🟢 Calculate shape directly from screen deltas: (ScreenPoint - FirstScreenPoint) / Zoom
        // This preserves perfect smoothness regardless of world coordinates.
        let zoom = zoomAtCreation
        let cosAngle = Double(cos(-rotationAngle))  // Negative to un-rotate
        let sinAngle = Double(sin(-rotationAngle))

        let relativePoints: [SIMD2<Float>] = screenPoints.map { pt in
            // Screen-space delta (high precision)
            let dx = Double(pt.x) - Double(firstScreenPt.x)
            let dy = Double(pt.y) - Double(firstScreenPt.y)

            // Un-rotate if needed (to match world space orientation)
            let unrotatedX = dx * cosAngle - dy * sinAngle
            let unrotatedY = dx * sinAngle + dy * cosAngle

            // Convert to world units by dividing by zoom
            let worldDx = unrotatedX / zoom
            let worldDy = unrotatedY / zoom

            return SIMD2<Float>(Float(worldDx), Float(worldDy))
        }

        // 3. World width is simply screen width (10px) divided by zoom
        let worldWidth = 10.0 / zoom
        self.worldWidth = worldWidth

        // 4. Tessellate in LOCAL space (no view-specific transforms)
        self.localVertices = tessellateStrokeLocal(
            centerPoints: relativePoints,
            width: Float(worldWidth)
        )
    }
}
