// BrushSettings.swift stores the current brush configuration (size, color, etc.)
// for use across the drawing system.
import Foundation
import SwiftUI
import simd

class BrushSettings: ObservableObject {
    @Published var size: Double = 10.0  // Stroke width in world units (1-100)
    @Published var color: SIMD4<Float> = SIMD4<Float>(1.0, 0.0, 0.0, 1.0)  // Red default

    static let minSize: Double = 1.0
    static let maxSize: Double = 100.0
}
