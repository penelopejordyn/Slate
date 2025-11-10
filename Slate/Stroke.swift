//
//  Stroke.swift
//  Slate
//
//  Created by Penny Marshall on 10/28/25.
//

import SwiftUI

struct Stroke {
    let id: UUID
    let centerPoints: [CGPoint]
    let width: CGFloat
    let color: SIMD4<Float>
    let origin: SIMD2<Float>
    let localVertices: [SIMD2<Float>]

    init(centerPoints: [CGPoint], width: CGFloat, color: SIMD4<Float>, viewSize _: CGSize) {
        self.id = UUID()
        self.centerPoints = centerPoints
        self.width = width
        self.color = color

        let mesh = tessellateStroke(
            centerPoints: centerPoints,
            width: width
        )
        self.origin = mesh.origin
        self.localVertices = mesh.localVertices
    }
}
