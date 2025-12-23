// BrushSettings.swift stores the current brush configuration (size, color, etc.)
// for use across the drawing system.
import Foundation
import SwiftUI
import simd

class BrushSettings: ObservableObject {
    enum ToolMode {
        case paint
        case maskEraser
        case strokeEraser
        case lasso
    }

    @Published var size: Double = 5.0  // Stroke width in world units (1-100)
    @Published var color: SIMD4<Float> = SIMD4<Float>(1.0, 0.0, 0.0, 1.0)  // Red default
    @Published var toolMode: ToolMode = .paint
    @Published var cullingMultiplier: Double = 1.0  // Test multiplier for culling box size (1.0, 0.5, 0.25)
    @Published var depthWriteEnabled: Bool = true  // Set false for strokes that should not write depth (e.g. translucent marker)
    @Published var constantScreenSize: Bool = true  // When true, stroke width is divided by zoom to stay constant on screen; when false, stroke scales with zoom

    static let minSize: Double = 5.0
    static let maxSize: Double = 1000.0

    var isMaskEraser: Bool { toolMode == .maskEraser }
    var isStrokeEraser: Bool { toolMode == .strokeEraser }
    var isLasso: Bool { toolMode == .lasso }
}
