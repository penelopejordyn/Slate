//
//  Stroke.swift
//  Slate
//
//  Created by Penny Marshall on 10/28/25.
//

import SwiftUI

struct StrokeVertex {
    var high: SIMD2<Float>
    var low: SIMD2<Float>
}

struct Stroke {
    let id: UUID
    let centerPoints: [CGPoint]
    let width: CGFloat
    let color: SIMD4<Float>
    let vertices: [StrokeVertex]  // ← Pre-tessellated in world space

    init(centerPoints: [CGPoint], width: CGFloat, color: SIMD4<Float>) {
        self.id = UUID()
        self.centerPoints = centerPoints
        self.width = width
        self.color = color

        // Tessellate once in world space
        self.vertices = tessellateStroke(
            centerPoints: centerPoints,
            width: width
        )
    }
}
