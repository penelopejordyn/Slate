// MetalCoordinator.swift manages the Metal pipeline, render passes, camera state,
// and gesture-driven updates for the drawing experience.
import Foundation
import Metal
import MetalKit
import simd

// MARK: - Coordinator

	class Coordinator: NSObject, MTKViewDelegate {
	    var device: MTLDevice!
	    var commandQueue: MTLCommandQueue!
	    var pipelineState: MTLRenderPipelineState!
	    var cardPipelineState: MTLRenderPipelineState!      // Pipeline for textured cards
	    var cardSolidPipelineState: MTLRenderPipelineState! // Pipeline for solid color cards
	    var cardLinedPipelineState: MTLRenderPipelineState! // Pipeline for lined paper cards
	    var cardGridPipelineState: MTLRenderPipelineState!  // Pipeline for grid paper cards
	    var cardShadowPipelineState: MTLRenderPipelineState! // Pipeline for card shadows
	    var strokeSegmentPipelineState: MTLRenderPipelineState! // SDF segment pipeline
	    var postProcessPipelineState: MTLRenderPipelineState!   // FXAA fullscreen pass
	    var samplerState: MTLSamplerState!                  // Sampler for card textures
	    var vertexBuffer: MTLBuffer!
	    var quadVertexBuffer: MTLBuffer!                    // Unit quad for instanced segments

    // Note: ICB removed - using simple GPU-offset approach instead

    //  Stencil States for Card Clipping
    var stencilStateDefault: MTLDepthStencilState! // Default passthrough (no testing)
    var stencilStateWrite: MTLDepthStencilState!   // Writes 1s to stencil (card background)
    var stencilStateRead: MTLDepthStencilState!    // Stencil read only (no depth test)
    var stencilStateClear: MTLDepthStencilState!   // Writes 0s to stencil (cleanup)

    // Depth States for Stroke Rendering
    var strokeDepthStateWrite: MTLDepthStencilState!   // Depth test + write enabled
    var strokeDepthStateNoWrite: MTLDepthStencilState! // Depth test enabled, depth write disabled
    var cardStrokeDepthStateWrite: MTLDepthStencilState!   // Depth test + write enabled, stencil read
    var cardStrokeDepthStateNoWrite: MTLDepthStencilState! // Depth test enabled, stencil read only

    // Offscreen render targets (scene -> texture, then FXAA -> drawable)
    private var offscreenColorTexture: MTLTexture?
    private var offscreenDepthStencilTexture: MTLTexture?
    private var offscreenTextureWidth: Int = 0
    private var offscreenTextureHeight: Int = 0

    // MARK: - Modal Input: Pencil vs. Finger

    /// Tracks what object we are currently drawing on with the pencil
    /// Now includes the Frame to support cross-depth drawing
    enum DrawingTarget {
        case canvas(Frame)
        case card(Card, Frame) // Track BOTH Card and the Frame it belongs to
    }
    var currentDrawingTarget: DrawingTarget?

    // MARK: - Lasso Selection
    struct LassoFrameSelection {
        var frame: Frame
        var strokeIDs: Set<UUID>
    }

    struct LassoCardSelection {
        let card: Card
        let frame: Frame
    }

    struct LassoCardStrokeSelection {
        let card: Card
        let frame: Frame
        let strokeIDs: Set<UUID>
    }

    struct LassoSelection {
        var points: [SIMD2<Double>] // Closed polygon in active frame coordinates
        var bounds: CGRect
        var center: SIMD2<Double>
        var frames: [LassoFrameSelection]
        var cards: [LassoCardSelection]
        var cardStrokes: [LassoCardStrokeSelection]
    }

    private struct StrokeSnapshot {
        let id: UUID
        let frame: Frame
        let index: Int
        let activePoints: [SIMD2<Double>]
        let color: SIMD4<Float>
        let worldWidth: Double
        let zoomEffectiveAtCreation: Float
        let depthID: UInt32
        let depthWriteEnabled: Bool
        let frameScale: Double
        let frameTranslation: SIMD2<Double>
    }

    private struct CardSnapshot {
        let card: Card
        let frameScale: Double
        let frameTranslation: SIMD2<Double>
        let originActive: SIMD2<Double>
        let size: SIMD2<Double>
        let rotation: Float
    }

    private struct CardStrokeSnapshot {
        let card: Card
        let frame: Frame
        let index: Int
        let activePoints: [SIMD2<Double>]
        let color: SIMD4<Float>
        let worldWidth: Double
        let zoomEffectiveAtCreation: Float
        let depthID: UInt32
        let depthWriteEnabled: Bool
        let frameScale: Double
        let frameTranslation: SIMD2<Double>
        let cardOrigin: SIMD2<Double>
        let cardRotation: Double
    }

    private struct LassoTransformState {
        let basePoints: [SIMD2<Double>]
        let baseCenter: SIMD2<Double>
        let baseStrokes: [StrokeSnapshot]
        let baseCards: [CardSnapshot]
        let baseCardStrokes: [CardStrokeSnapshot]
        var currentScale: Double
        var currentRotation: Double
    }

    private enum LassoTarget {
        case canvas
        case card(Card, Frame)
    }

    // MARK: - Undo/Redo System

    enum UndoAction {
        case drawStroke(stroke: Stroke, target: DrawingTarget)
        case eraseStroke(stroke: Stroke, strokeIndex: Int, target: DrawingTarget)
        case moveCard(card: Card, frame: Frame, oldOrigin: SIMD2<Double>)
        case resizeCard(card: Card, frame: Frame, oldOrigin: SIMD2<Double>, oldSize: SIMD2<Double>)
        // TODO: Add lasso move/transform cases
    }

    private var undoStack: [UndoAction] = []
    private var redoStack: [UndoAction] = []
    private let maxUndoActions = 25

    var currentTouchPoints: [CGPoint] = []      // Real historical points (SCREEN space)
    var predictedTouchPoints: [CGPoint] = []    // Future points (Transient, SCREEN space)
    var liveStrokeOrigin: SIMD2<Double>?        // Temporary origin for live stroke (Double precision)

    var lassoDrawingPoints: [CGPoint] = []
    var lassoPredictedPoints: [CGPoint] = []
    var lassoSelection: LassoSelection?
    var lassoPreviewStroke: Stroke?
    var lassoPreviewFrame: Frame?
    private var lassoTransformState: LassoTransformState?
    private var lassoTarget: LassoTarget?
    private var lassoPreviewCard: Card?
    private var lassoPreviewCardFrame: Frame?

    //  OPTIMIZATION: Adaptive Fidelity
    // Track last saved point for distance filtering to prevent vertex explosion during slow drawing
    var lastSavedPoint: CGPoint?

    // Telescoping Reference Frames
    // Instead of a flat array, we use a linked list of Frames for infinite zoom
    var rootFrame = Frame()           // The "Base Reality" - top level that cannot be zoomed out of
    lazy var activeFrame: Frame = rootFrame  // The current "Local Universe" we are viewing/editing

    weak var metalView: MTKView?

    // MARK: - Card Interaction Callbacks
    var onEditCard: ((Card) -> Void)?

    //  UPGRADED: Store camera state as Double for infinite precision
    var panOffset: SIMD2<Double> = .zero
    var zoomScale: Double = 1.0
    var rotationAngle: Float = 0.0

    // MARK: - Brush Settings
    let brushSettings = BrushSettings()

    private let lassoDashLengthPx: Double = 6.0
    private let lassoGapLengthPx: Double = 4.0
    private let lassoLineWidthPx: Double = 1.5
    private let lassoColor = SIMD4<Float>(1.0, 1.0, 1.0, 0.6)
    private let cardCornerRadiusPx: Float = 12.0
    private let cardShadowEnabled: Bool = true
    private let cardShadowBlurPx: Float = 18.0
    private let cardShadowOpacity: Float = 0.25
    private let cardShadowOffsetPx = SIMD2<Float>(0.0, 0.0)

	    // MARK: - Debug Metrics
	    var debugDrawnVerticesThisFrame: Int = 0
	    var debugDrawnNodesThisFrame: Int = 0

	    // MARK: - Debug Tools
	    func debugPopulateFrames(parentCount: Int = 20,
	                             childCount: Int = 20,
	                             strokesPerFrame: Int = 1000,
	                             maxOffset: Double = 1000.0) {
	        guard let view = metalView else { return }

	        let cameraCenterActive = calculateCameraCenterWorld(viewSize: view.bounds.size)

	        // Build parent chain above the current top-most frame (linked-list invariant).
	        var topFrame = activeFrame
	        while let parent = topFrame.parent {
	            topFrame = parent
	        }

	        if parentCount > 0 {
	            for _ in 0..<parentCount {
	                let newParent = Frame(depth: topFrame.depthFromRoot - 1)
	                topFrame.parent = newParent
	                newParent.children.append(topFrame)
	                topFrame.originInParent = .zero
	                topFrame.scaleRelativeToParent = 1000.0
	                topFrame = newParent
	            }
	        }

	        // Build child chain below the current bottom-most frame (linked-list invariant).
	        var bottomFrame = activeFrame
	        while let child = childFrame(of: bottomFrame) {
	            bottomFrame = child
	        }

	        if childCount > 0 {
	            var parentFrame = bottomFrame
	            for _ in 0..<childCount {
	                let parentCameraCenter = cameraCenterInFrame(parentFrame, cameraCenterActive: cameraCenterActive)
	                let newChild = Frame(parent: parentFrame,
	                                     origin: parentCameraCenter,
	                                     scale: 1000.0,
	                                     depth: parentFrame.depthFromRoot + 1)
	                parentFrame.children.append(newChild)
	                parentFrame = newChild
	            }
	        }

	        // Debug stress fill: keep strokes within +/- maxOffset in each frame's world space.
	        let transforms = collectFrameTransforms(pointActive: cameraCenterActive)
	        var frames: [Frame] = transforms.ancestors.map { $0.frame }
	        frames.append(activeFrame)
	        frames.append(contentsOf: transforms.descendants.map { $0.frame })

	        for frame in frames {
	            let (cameraCenterInTarget, effectiveZoom) = cameraCenterAndZoom(in: frame, cameraCenterActive: cameraCenterActive)
	            frame.strokes.reserveCapacity(frame.strokes.count + strokesPerFrame)

	            for _ in 0..<strokesPerFrame {
	                let points = randomStrokePoints(center: cameraCenterInTarget, maxOffset: maxOffset)
	                let virtualScreenPoints = points.map { CGPoint(x: $0.x * effectiveZoom, y: $0.y * effectiveZoom) }
	                let color = SIMD4<Float>(Float.random(in: 0.1...1.0),
	                                         Float.random(in: 0.1...1.0),
	                                         Float.random(in: 0.1...1.0),
	                                         1.0)
	                let stroke = Stroke(screenPoints: virtualScreenPoints,
	                                    zoomAtCreation: effectiveZoom,
	                                    panAtCreation: .zero,
	                                    viewSize: .zero,
	                                    rotationAngle: 0,
	                                    color: color,
	                                    baseWidth: Double.random(in: 2.0...10.0),
	                                    zoomEffectiveAtCreation: Float(effectiveZoom),
	                                    device: device,
	                                    depthID: allocateStrokeDepthID(),
	                                    depthWriteEnabled: true)
	                frame.strokes.append(stroke)
	            }
	        }
	    }

	    func clearAllStrokes() {
	        var topFrame = activeFrame
	        while let parent = topFrame.parent {
	            topFrame = parent
	        }

	        var current: Frame? = topFrame
	        while let frame = current {
	            frame.strokes.removeAll()
	            for card in frame.cards {
	                card.strokes.removeAll()
	            }
	            current = childFrame(of: frame)
	        }
	    }

	    // MARK: - Undo/Redo Implementation

	    func pushUndo(_ action: UndoAction) {
	        undoStack.append(action)
	        if undoStack.count > maxUndoActions {
	            undoStack.removeFirst()
	        }
	        redoStack.removeAll() // Clear redo stack when new action is performed
	    }

	    func pushUndoMoveCard(card: Card, frame: Frame, oldOrigin: SIMD2<Double>) {
	        pushUndo(.moveCard(card: card, frame: frame, oldOrigin: oldOrigin))
	    }

	    func pushUndoResizeCard(card: Card, frame: Frame, oldOrigin: SIMD2<Double>, oldSize: SIMD2<Double>) {
	        pushUndo(.resizeCard(card: card, frame: frame, oldOrigin: oldOrigin, oldSize: oldSize))
	    }

	    // TODO: Implement lasso snapshot capture for undo support

	    func undo() {
	        guard let action = undoStack.popLast() else { return }

	        switch action {
	        case .drawStroke(let stroke, let target):
	            // Remove the stroke
	            switch target {
	            case .canvas(let frame):
	                if let index = frame.strokes.firstIndex(where: { $0.id == stroke.id }) {
	                    frame.strokes.remove(at: index)
	                }
	            case .card(let card, _):
	                if let index = card.strokes.firstIndex(where: { $0.id == stroke.id }) {
	                    card.strokes.remove(at: index)
	                }
	            }
	            redoStack.append(action)

	        case .eraseStroke(let stroke, let strokeIndex, let target):
	            // Restore the stroke
	            switch target {
	            case .canvas(let frame):
	                frame.strokes.insert(stroke, at: min(strokeIndex, frame.strokes.count))
	            case .card(let card, _):
	                card.strokes.insert(stroke, at: min(strokeIndex, card.strokes.count))
	            }
	            redoStack.append(action)

	        case .moveCard(let card, _, let oldOrigin):
	            let currentOrigin = card.origin
	            card.origin = oldOrigin
	            redoStack.append(.moveCard(card: card, frame: activeFrame, oldOrigin: currentOrigin))

	        case .resizeCard(let card, _, let oldOrigin, let oldSize):
	            let currentOrigin = card.origin
	            let currentSize = card.size
	            card.origin = oldOrigin
	            card.size = oldSize
	            card.rebuildGeometry()
	            redoStack.append(.resizeCard(card: card, frame: activeFrame, oldOrigin: currentOrigin, oldSize: currentSize))
	        }
	    }

	    func redo() {
	        guard let action = redoStack.popLast() else { return }

	        switch action {
	        case .drawStroke(let stroke, let target):
	            // Re-add the stroke
	            switch target {
	            case .canvas(let frame):
	                frame.strokes.append(stroke)
	            case .card(let card, _):
	                card.strokes.append(stroke)
	            }
	            undoStack.append(action)

	        case .eraseStroke(let stroke, let strokeIndex, let target):
	            // Re-remove the stroke
	            switch target {
	            case .canvas(let frame):
	                if let index = frame.strokes.firstIndex(where: { $0.id == stroke.id }) {
	                    frame.strokes.remove(at: index)
	                }
	            case .card(let card, _):
	                if let index = card.strokes.firstIndex(where: { $0.id == stroke.id }) {
	                    card.strokes.remove(at: index)
	                }
	            }
	            undoStack.append(action)

	        case .moveCard(let card, let frame, let oldOrigin):
	            let currentOrigin = card.origin
	            card.origin = oldOrigin
	            undoStack.append(.moveCard(card: card, frame: frame, oldOrigin: currentOrigin))

	        case .resizeCard(let card, let frame, let oldOrigin, let oldSize):
	            let currentOrigin = card.origin
	            let currentSize = card.size
	            card.origin = oldOrigin
	            card.size = oldSize
	            card.rebuildGeometry()
	            undoStack.append(.resizeCard(card: card, frame: frame, oldOrigin: currentOrigin, oldSize: currentSize))
	        }
	    }

	    func replaceCanvas(with newRoot: Frame) {
	        newRoot.parent = nil
	        rootFrame = newRoot
	        activeFrame = findFrame(withDepth: 0, in: newRoot) ?? newRoot
	        panOffset = .zero
	        zoomScale = 1.0
	        rotationAngle = 0.0
	        currentDrawingTarget = nil
	        currentTouchPoints = []
	        predictedTouchPoints = []
	        liveStrokeOrigin = nil
	        lastSavedPoint = nil
	        clearLassoSelection()
	        lassoDrawingPoints = []
	        lassoPredictedPoints = []
	        resetStrokeDepthID(using: newRoot)
	    }

	    // MARK: - Global Stroke Depth Ordering
	    // A monotonic per-stroke counter lets depth testing work across telescoping frames.
	    // Larger depthID = newer stroke; we map this into Metal NDC depth (smaller = closer).
	    private static let strokeDepthSlotCount: UInt32 = 1 << 24
	    private static let strokeDepthDenominator: Float = Float(strokeDepthSlotCount) + 1.0
	    private var nextStrokeDepthID: UInt32 = 0

	    private func allocateStrokeDepthID() -> UInt32 {
	        let id = nextStrokeDepthID
	        if nextStrokeDepthID < Self.strokeDepthSlotCount - 1 {
	            nextStrokeDepthID += 1
	        }
	        return id
	    }

	    private func peekStrokeDepthID() -> UInt32 {
	        nextStrokeDepthID
	    }

	    private func resetStrokeDepthID(using frame: Frame) {
	        if let maxDepth = maxStrokeDepthID(in: frame) {
	            if maxDepth < Self.strokeDepthSlotCount - 1 {
	                nextStrokeDepthID = maxDepth + 1
	            } else {
	                nextStrokeDepthID = Self.strokeDepthSlotCount - 1
	            }
	        } else {
	            nextStrokeDepthID = 0
	        }
	    }

	    private func maxStrokeDepthID(in frame: Frame) -> UInt32? {
	        var maxID: UInt32?

	        func consider(_ id: UInt32) {
	            if let current = maxID {
	                if id > current {
	                    maxID = id
	                }
	            } else {
	                maxID = id
	            }
	        }

	        for stroke in frame.strokes {
	            consider(stroke.depthID)
	        }
	        for card in frame.cards {
	            for stroke in card.strokes {
	                consider(stroke.depthID)
	            }
	        }
	        for child in frame.children {
	            if let childMax = maxStrokeDepthID(in: child) {
	                consider(childMax)
	            }
	        }

	        return maxID
	    }

	    private func findFrame(withDepth depth: Int, in frame: Frame) -> Frame? {
	        if frame.depthFromRoot == depth {
	            return frame
	        }
	        for child in frame.children {
	            if let found = findFrame(withDepth: depth, in: child) {
	                return found
	            }
	        }
	        return nil
	    }

	    override init() {
	        super.init()
	        device = MTLCreateSystemDefaultDevice()!
        commandQueue = device.makeCommandQueue()!

        makePipeLine()
        makeVertexBuffer()
        makeQuadVertexBuffer()
    }

	    // MARK: - Recursive Renderer
	    /// Calculate Metal depth value for a stroke based on creation order
	    ///
	    /// **DEPTH TESTING BY STROKE ORDER:**
	    /// Newer strokes (higher depthID) are always rendered on top of older strokes,
	    /// regardless of which telescope depth they were created at.
	    ///
	    /// Example:
	    /// - Draw stroke A at depth -9 (depthID = 100)
	    /// - Draw stroke B at depth 0  (depthID = 200)
	    /// - Draw stroke C at depth -9 (depthID = 300)
	    ///
	    /// Result: C is on top of B, which is on top of A
	    ///
	    /// This uses the global monotonic depthID counter, so stroke ordering is consistent
	    /// across all telescope depths from -∞ to +∞.
	    ///
	    /// **OVERDRAW PREVENTION:**
	    /// All segments within a stroke share the same depth value. The depth buffer prevents
	    /// pixel-level overdraw when drawing complex shapes. Front-to-back rendering of
	    /// depth-write-enabled strokes provides early-Z rejection for maximum performance.
	    ///
	    /// - Parameters:
	    ///   - depthID: The stroke's depth ID (monotonic counter for creation order)
	    /// - Returns: Metal NDC depth value [0, 1] where 0 is closest
	    private func strokeDepth(for depthID: UInt32) -> Float {
	        // Map depthID directly to depth buffer
	        // Higher depthID (newer stroke) → lower depth value (closer to camera)
	        let clamped = min(depthID, Self.strokeDepthSlotCount - 1)
	        let numerator = Float(Self.strokeDepthSlotCount - clamped)
	        return numerator / Self.strokeDepthDenominator
	    }

    /// Recursively render a frame and adjacent depth levels (depth ±1).
    ///
    /// ** BIDIRECTIONAL RENDERING:**
    /// We now render in three layers:
    /// 1. Parent frame (background - depth -1)
    /// 2. Current frame (middle layer - depth 0)
    /// 3. Child frames (foreground details - depth +1)
    ///
    /// This ensures strokes remain visible when transitioning between depths.
    ///
    /// - Parameters:
    ///   - frame: The frame to render
    ///   - cameraCenterInThisFrame: Where the camera is positioned in this frame's coordinate system
    ///   - viewSize: The view dimensions
    ///   - currentZoom: The current zoom level (adjusted for each frame level)
    ///   - currentRotation: The rotation angle
    ///   - encoder: The Metal render encoder
    func renderFrame(_ frame: Frame,
                     cameraCenterInThisFrame: SIMD2<Double>,
                     viewSize: CGSize,
                     currentZoom: Double,
                     currentRotation: Float,
                     encoder: MTLRenderCommandEncoder,
                     excludedChild: Frame? = nil,
                     depthFromActive: Int = 0) { //  NEW: Prevent double-rendering

        // Reset debug metrics at the start of each frame (only for root call)
        if frame === activeFrame && excludedChild == nil {
            debugDrawnVerticesThisFrame = 0
            debugDrawnNodesThisFrame = 0
        }

        // LAYER 1: RENDER PARENT (Background - Depth -1) -------------------------------
        if let parent = frame.parent {
            // Prevent recursion loops / redundant overdraw when traversing from parent -> child.
            let cameFromParent = excludedChild.map { $0 === parent } ?? false
            if !cameFromParent {
            // Convert camera position from child coordinates to parent coordinates
            // Formula: parent_pos = originInParent + (child_pos / scale)
            let cameraCenterInParent = SIMD2<Double>(
                frame.originInParent.x + (cameraCenterInThisFrame.x / frame.scaleRelativeToParent),
                frame.originInParent.y + (cameraCenterInThisFrame.y / frame.scaleRelativeToParent)
            )

            // Zoom in parent frame is reduced (parent is "bigger")
            let parentZoom = currentZoom * frame.scaleRelativeToParent

            //  FIX: Event Horizon Culling - Lowered to 1e9 (1 Billion)
            // Stop rendering the parent if it's magnified beyond Double precision safe zone.
            // At 1e9+, precision errors cause jittery/shakey panning across vast distances.
            // The background becomes mathematically unstable and visually meaningless.
            if parentZoom > 0.0001 && parentZoom < 1e9 {
                renderFrame(parent,
                           cameraCenterInThisFrame: cameraCenterInParent,
                           viewSize: viewSize,
                           currentZoom: parentZoom,
                           currentRotation: currentRotation,
                           encoder: encoder,
                           excludedChild: frame,
                           depthFromActive: depthFromActive - 1) //  TELL PARENT TO SKIP US
            }
            }
        }

        // LAYER 2: RENDER THIS FRAME (Middle Layer - Depth 0) --------------------------

        // 2.1: RENDER CANVAS STROKES (Background layer - below cards)
        // SCREEN SPACE CULLING FIX:
        // Instead of calculating world bounds (which fail at extreme zoom), we calculate
        // the "Maximum Visible Radius" from the screen center in screen-space pixels.
        // This is numerically stable at all zoom levels because screen dimensions are constant.

        let screenW = Double(viewSize.width)
        let screenH = Double(viewSize.height)

        // Distance from screen center to corner (diagonal)
        let screenRadius = sqrt(screenW * screenW + screenH * screenH) * 0.5

        // Apply Culling Multiplier for testing
        let cullRadius = screenRadius * brushSettings.cullingMultiplier

        // Render canvas strokes using depth testing (early-Z).
        encoder.setRenderPipelineState(strokeSegmentPipelineState)
        encoder.setCullMode(.none)
        encoder.setVertexBuffer(quadVertexBuffer, offset: 0, index: 0)

        let zoom = max(currentZoom, 1e-6)
        let angle = Double(currentRotation)
        let c = cos(angle)
        let s = sin(angle)

        drawLiveEraserOnCanvasIfNeeded(frame: frame,
                                       cameraCenterInThisFrame: cameraCenterInThisFrame,
                                       viewSize: viewSize,
                                       currentZoom: currentZoom,
                                       currentRotation: currentRotation,
                                       encoder: encoder)

	        let strokeCount = frame.strokes.count
	        func depthForStroke(_ stroke: Stroke) -> Float {
	            strokeDepth(for: stroke.depthID)
	        }

        func drawStroke(_ stroke: Stroke, depth: Float) {
            guard !stroke.segments.isEmpty, let segmentBuffer = stroke.segmentBuffer else { return }

            // ZOOM-BASED CULLING: Skip strokes drawn at extreme zoom when we're zoomed way out
            // If current zoom is more than 100,000x higher than when the stroke was created,
            // the stroke would be invisible (sub-pixel), so skip all math and rendering
            let strokeZoom = max(Double(stroke.zoomEffectiveAtCreation), 1.0) // Treat 0 as 1
            if zoom > strokeZoom * 100_000.0 {
                return
            }

            let dx = stroke.origin.x - cameraCenterInThisFrame.x
            let dy = stroke.origin.y - cameraCenterInThisFrame.y

            // Screen-space culling (stable at extreme zoom levels).
            let distWorld = sqrt(dx * dx + dy * dy)
            let distScreen = distWorld * zoom

            let bounds = stroke.localBounds
            let farX = max(abs(Double(bounds.minX)), abs(Double(bounds.maxX)))
            let farY = max(abs(Double(bounds.minY)), abs(Double(bounds.maxY)))
            let radiusWorld = sqrt(farX * farX + farY * farY)
            let radiusScreen = radiusWorld * zoom

            if (distScreen - radiusScreen) > cullRadius { return }

            let rotatedOffsetScreen = SIMD2<Float>(
                Float((dx * c - dy * s) * zoom),
                Float((dx * s + dy * c) * zoom)
            )

            let basePixelWidth = Float(stroke.worldWidth * zoom)
            let halfPixelWidth = max(basePixelWidth * 0.5, 0.5)

            var transform = StrokeTransform(
                relativeOffset: .zero,
                rotatedOffsetScreen: rotatedOffsetScreen,
                zoomScale: Float(zoom),
                screenWidth: Float(viewSize.width),
                screenHeight: Float(viewSize.height),
                rotationAngle: currentRotation,
                halfPixelWidth: halfPixelWidth,
                featherPx: 1.0,
                depth: depth
            )

            encoder.setVertexBytes(&transform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
            encoder.setFragmentBytes(&transform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
            encoder.setVertexBuffer(segmentBuffer, offset: 0, index: 2)
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: stroke.segments.count)
        }

	        // Depth-write strokes: newest -> oldest (front-to-back).
	        encoder.setDepthStencilState(strokeDepthStateWrite)
	        if strokeCount > 0 {
	            for i in stride(from: strokeCount - 1, through: 0, by: -1) {
	                let stroke = frame.strokes[i]
	                guard stroke.depthWriteEnabled else { continue }
	                drawStroke(stroke, depth: depthForStroke(stroke))
	            }
	        }

        // No-depth-write strokes: oldest -> newest (painter's algorithm), but depth-tested.
	        encoder.setDepthStencilState(strokeDepthStateNoWrite)
	        for i in 0..<strokeCount {
	            let stroke = frame.strokes[i]
	            guard !stroke.depthWriteEnabled else { continue }
	            drawStroke(stroke, depth: depthForStroke(stroke))
	        }

        // 2.2: RENDER CARDS (Middle layer - on top of canvas strokes)
        for card in frame.cards {
            // A. Calculate Position
            // Card lives in the Frame, so it moves with the Frame
            let relativeOffsetDouble = card.origin - cameraCenterInThisFrame
            let cardOffsetLocal = cardOffsetFromCameraInLocalSpace(card: card,
                                                                   cameraCenter: cameraCenterInThisFrame)

            // SCREEN SPACE CULLING for cards
            let distWorld = sqrt(relativeOffsetDouble.x * relativeOffsetDouble.x + relativeOffsetDouble.y * relativeOffsetDouble.y)
            let distScreen = distWorld * currentZoom
            let cardRadiusWorld = sqrt(pow(card.size.x, 2) + pow(card.size.y, 2)) * 0.5
            let cardRadiusScreen = cardRadiusWorld * currentZoom

            if (distScreen - cardRadiusScreen) > cullRadius {
                continue // Cull card
            }

            let relativeOffset = SIMD2<Float>(Float(cardOffsetLocal.x), Float(cardOffsetLocal.y))
            let cardHalfSize = SIMD2<Float>(Float(card.size.x * 0.5), Float(card.size.y * 0.5))
            var style = CardStyleUniforms(
                cardHalfSize: cardHalfSize,
                zoomScale: Float(currentZoom),
                cornerRadiusPx: cardCornerRadiusPx,
                shadowBlurPx: cardShadowBlurPx,
                shadowOpacity: cardShadowOpacity,
                cardOpacity: card.opacity
            )

            // B. Handle Rotation
            // Cards have their own rotation property
            // Total Rotation = Camera Rotation + Card Rotation
            let finalRotation = currentRotation + card.rotation

            var transform = CardTransform(
                relativeOffset: relativeOffset,
                zoomScale: Float(currentZoom),
                screenWidth: Float(viewSize.width),
                screenHeight: Float(viewSize.height),
                rotationAngle: finalRotation,
                depth: 1.0
            )

            if cardShadowEnabled {
                let shadowExpandWorld = Double(cardShadowBlurPx) / max(currentZoom, 1e-6)
                let scaleX = (card.size.x * 0.5 + shadowExpandWorld) / max(card.size.x * 0.5, 1e-6)
                let scaleY = (card.size.y * 0.5 + shadowExpandWorld) / max(card.size.y * 0.5, 1e-6)
                let shadowVertices = card.localVertices.map { vertex in
                    StrokeVertex(
                        position: SIMD2<Float>(vertex.position.x * Float(scaleX), vertex.position.y * Float(scaleY)),
                        uv: vertex.uv,
                        color: vertex.color
                    )
                }

                let shadowOffset = shadowOffsetInCardLocalSpace(offsetPx: cardShadowOffsetPx,
                                                                rotation: finalRotation,
                                                                zoom: currentZoom)
                var shadowTransform = CardTransform(
                    relativeOffset: SIMD2<Float>(Float(cardOffsetLocal.x + shadowOffset.x),
                                                 Float(cardOffsetLocal.y + shadowOffset.y)),
                    zoomScale: Float(currentZoom),
                    screenWidth: Float(viewSize.width),
                    screenHeight: Float(viewSize.height),
                    rotationAngle: finalRotation,
                    depth: 1.0
                )

                if let shadowBuffer = device.makeBuffer(bytes: shadowVertices,
                                                        length: shadowVertices.count * MemoryLayout<StrokeVertex>.stride,
                                                        options: .storageModeShared) {
                    encoder.setDepthStencilState(stencilStateDefault)
                    encoder.setRenderPipelineState(cardShadowPipelineState)
                    encoder.setVertexBytes(&shadowTransform, length: MemoryLayout<CardTransform>.stride, index: 1)
                    encoder.setFragmentBytes(&style, length: MemoryLayout<CardStyleUniforms>.stride, index: 2)
                    encoder.setVertexBuffer(shadowBuffer, offset: 0, index: 0)
                    encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
                }
            }

            //  STEP 1: DRAW CARD BACKGROUND + WRITE STENCIL
            // Write '1' into the stencil buffer where the card pixels are
            encoder.setDepthStencilState(stencilStateWrite)
            encoder.setStencilReferenceValue(1)

            // C. Set Pipeline & Bind Content Based on Card Type
            switch card.type {
            case .solidColor:
                // Use solid color pipeline (no texture required)
                encoder.setRenderPipelineState(cardSolidPipelineState)
                encoder.setVertexBytes(&transform, length: MemoryLayout<CardTransform>.stride, index: 1)
                var c = card.backgroundColor
                encoder.setFragmentBytes(&c, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
                encoder.setFragmentBytes(&style, length: MemoryLayout<CardStyleUniforms>.stride, index: 2)

            case .image(let texture):
                // Use textured pipeline (requires texture binding)
                encoder.setRenderPipelineState(cardPipelineState)
                encoder.setVertexBytes(&transform, length: MemoryLayout<CardTransform>.stride, index: 1)
                encoder.setFragmentTexture(texture, index: 0)
                encoder.setFragmentSamplerState(samplerState, index: 0)
                encoder.setFragmentBytes(&style, length: MemoryLayout<CardStyleUniforms>.stride, index: 2)

            case .lined(let config):
                // Use procedural lined paper pipeline
                encoder.setRenderPipelineState(cardLinedPipelineState)
                encoder.setVertexBytes(&transform, length: MemoryLayout<CardTransform>.stride, index: 1)

                // 1. Background Color
                var bg = card.backgroundColor
                encoder.setFragmentBytes(&bg, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)

                // 2. Uniforms (Lines)
                // FIX: Calculate World Spacing based on Creation Zoom
                // Formula: SpacingPts / CreationZoom = SpacingWorld
                // Example: 25pts / 1000x = 0.025 world units
                let worldSpacing = config.spacing / Float(card.creationZoom)
                let worldLineWidth = config.lineWidth / Float(card.creationZoom)

                var uniforms = CardShaderUniforms(
                    spacing: worldSpacing,        // Pass WORLD units to shader
                    lineWidth: worldLineWidth,    // Scale line width too!
                    color: config.color,
                    cardWidth: Float(card.size.x),
                    cardHeight: Float(card.size.y)
                )
                encoder.setFragmentBytes(&uniforms, length: MemoryLayout<CardShaderUniforms>.stride, index: 1)
                encoder.setFragmentBytes(&style, length: MemoryLayout<CardStyleUniforms>.stride, index: 2)

            case .grid(let config):
                // Use procedural grid paper pipeline
                encoder.setRenderPipelineState(cardGridPipelineState)
                encoder.setVertexBytes(&transform, length: MemoryLayout<CardTransform>.stride, index: 1)

                // 1. Background Color
                var bg = card.backgroundColor
                encoder.setFragmentBytes(&bg, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)

                // 2. Uniforms (Grid)
                // FIX: Calculate World Spacing based on Creation Zoom
                let worldSpacing = config.spacing / Float(card.creationZoom)
                let worldLineWidth = config.lineWidth / Float(card.creationZoom)

                var uniforms = CardShaderUniforms(
                    spacing: worldSpacing,        // Pass WORLD units to shader
                    lineWidth: worldLineWidth,    // Scale line width too!
                    color: config.color,
                    cardWidth: Float(card.size.x),
                    cardHeight: Float(card.size.y)
                )
                encoder.setFragmentBytes(&uniforms, length: MemoryLayout<CardShaderUniforms>.stride, index: 1)
                encoder.setFragmentBytes(&style, length: MemoryLayout<CardStyleUniforms>.stride, index: 2)

            case .drawing:
                continue // Future: Render nested strokes
            }

            // D. Draw the Card Quad (writes to both color buffer and stencil)
            let vertexBuffer = device.makeBuffer(
                bytes: card.localVertices,
                length: card.localVertices.count * MemoryLayout<StrokeVertex>.stride,
                options: .storageModeShared
            )
            encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)

            //  STEP 2: DRAW CARD STROKES (CLIPPED TO CARD)
            // Only draw where stencil == 1
            let isLiveCardEraserTarget: Bool
            if brushSettings.isMaskEraser, let target = currentDrawingTarget,
               case .card(let targetCard, let targetFrame) = target,
               targetFrame === frame, targetCard === card {
                isLiveCardEraserTarget = true
            } else {
                isLiveCardEraserTarget = false
            }

            let isLiveCardLassoTarget = (lassoPreviewCard === card && lassoPreviewCardFrame === frame)

            if !card.strokes.isEmpty || isLiveCardEraserTarget || isLiveCardLassoTarget {
                encoder.setRenderPipelineState(strokeSegmentPipelineState)
                encoder.setStencilReferenceValue(1)
                encoder.setVertexBuffer(quadVertexBuffer, offset: 0, index: 0)

                // Calculate "magic offset" for card-local coordinates
                let totalRotation = currentRotation + card.rotation
                let offset = SIMD2<Float>(Float(cardOffsetLocal.x), Float(cardOffsetLocal.y))

                func drawCardStroke(_ stroke: Stroke) {
                    guard !stroke.segments.isEmpty, let segmentBuffer = stroke.segmentBuffer else { return }

                    // ZOOM-BASED CULLING: Skip strokes drawn at extreme zoom when we're zoomed way out
                    let strokeZoom = max(Double(stroke.zoomEffectiveAtCreation), 1.0) // Treat 0 as 1
                    if currentZoom > strokeZoom * 100_000.0 {
                        return
                    }

                    let strokeOffset = stroke.origin
                    let strokeRelativeOffset = offset + SIMD2<Float>(Float(strokeOffset.x), Float(strokeOffset.y))

                    // Calculate screen-space thickness for card strokes
                    let basePixelWidth = Float(stroke.worldWidth * currentZoom)
                    let halfPixelWidth = max(basePixelWidth * 0.5, 0.5)

                    // Calculate depth based on stroke's depthID (creation order)
                    let cardStrokeDepth = strokeDepth(for: stroke.depthID)

                    var strokeTransform = StrokeTransform(
                        relativeOffset: strokeRelativeOffset,
                        rotatedOffsetScreen: SIMD2<Float>(Float((Double(strokeRelativeOffset.x) * cos(Double(totalRotation)) - Double(strokeRelativeOffset.y) * sin(Double(totalRotation))) * currentZoom),
                                                          Float((Double(strokeRelativeOffset.x) * sin(Double(totalRotation)) + Double(strokeRelativeOffset.y) * cos(Double(totalRotation))) * currentZoom)),
                        zoomScale: Float(currentZoom),
                        screenWidth: Float(viewSize.width),
                        screenHeight: Float(viewSize.height),
                        rotationAngle: totalRotation,
                        halfPixelWidth: halfPixelWidth,
                        featherPx: 1.0,
                        depth: cardStrokeDepth
                    )

                    encoder.setVertexBytes(&strokeTransform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
                    encoder.setFragmentBytes(&strokeTransform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
                    encoder.setVertexBuffer(segmentBuffer, offset: 0, index: 2)
                    encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: stroke.segments.count)
                }

                let liveCardStroke: Stroke?
                if isLiveCardEraserTarget, let screenPoints = buildLiveScreenPoints() {
                    liveCardStroke = createStrokeForCard(
                        screenPoints: screenPoints,
                        card: card,
                        frame: frame,
                        viewSize: viewSize,
                        depthID: peekStrokeDepthID(),
                        color: SIMD4<Float>(0, 0, 0, 0),
                        depthWriteEnabled: true
                    )
                } else {
                    liveCardStroke = nil
                }

                let liveCardLassoStroke = isLiveCardLassoTarget ? lassoPreviewStroke : nil

                let strokeCount = card.strokes.count

                // Depth-write strokes: newest -> oldest (front-to-back).
                encoder.setDepthStencilState(cardStrokeDepthStateWrite)
                if let liveStroke = liveCardStroke {
                    drawCardStroke(liveStroke)
                }
                if strokeCount > 0 {
                    for i in stride(from: strokeCount - 1, through: 0, by: -1) {
                        let stroke = card.strokes[i]
                        guard stroke.depthWriteEnabled else { continue }
                        drawCardStroke(stroke)
                    }
                }

                // No-depth-write strokes: oldest -> newest (painter's), but depth-tested.
                encoder.setDepthStencilState(cardStrokeDepthStateNoWrite)
                if let liveLasso = liveCardLassoStroke {
                    drawCardStroke(liveLasso)
                }
                for i in 0..<strokeCount {
                    let stroke = card.strokes[i]
                    guard !stroke.depthWriteEnabled else { continue }
                    drawCardStroke(stroke)
                }
            }

            //  STEP 3: CLEANUP STENCIL (Reset to 0 for next card)
            // Draw the card quad again with stencil clear mode
            encoder.setRenderPipelineState(cardSolidPipelineState)
            encoder.setDepthStencilState(stencilStateClear)
            encoder.setStencilReferenceValue(0)
            encoder.setVertexBytes(&transform, length: MemoryLayout<CardTransform>.stride, index: 1)
            var clearColor = SIMD4<Float>(0, 0, 0, 0)
            encoder.setFragmentBytes(&clearColor, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
            encoder.setFragmentBytes(&style, length: MemoryLayout<CardStyleUniforms>.stride, index: 2)
            encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)

            // E. Draw Resize Handles (If Selected) 
            // Handles should not be affected by stencil
            if card.isLocked {
                drawCardLockIcon(card: card,
                                 cameraCenter: cameraCenterInThisFrame,
                                 viewSize: viewSize,
                                 zoom: currentZoom,
                                 rotation: currentRotation,
                                 encoder: encoder)
            }
            if card.isEditing && !card.isLocked {
                encoder.setDepthStencilState(stencilStateDefault) // Disable stencil for handles
                drawCardHandles(card: card,
                                cameraCenter: cameraCenterInThisFrame,
                                viewSize: viewSize,
                                zoom: currentZoom,
                                rotation: currentRotation,
                                encoder: encoder)
            }
        }

        // Reset pipeline and stencil state for subsequent rendering (live stroke, child frames, etc.)
        encoder.setRenderPipelineState(strokeSegmentPipelineState)
        encoder.setDepthStencilState(stencilStateDefault)

        // LAYER 3: RENDER CHILDREN (Foreground Details - Depth +1) ---------------------
        for child in frame.children {
            if let excluded = excludedChild, child === excluded { continue }

            let childZoom = currentZoom / child.scaleRelativeToParent
            if childZoom < 0.001 { continue }

            let cameraCenterInChild = (cameraCenterInThisFrame - child.originInParent) * child.scaleRelativeToParent

            renderFrame(child,
                        cameraCenterInThisFrame: cameraCenterInChild,
                        viewSize: viewSize,
                        currentZoom: childZoom,
                        currentRotation: currentRotation,
                        encoder: encoder,
                        excludedChild: frame,
                        depthFromActive: depthFromActive + 1)
        }
    }

    private func buildLiveScreenPoints() -> [CGPoint]? {
        guard !currentTouchPoints.isEmpty else { return nil }

        var screenPoints = currentTouchPoints
        if let last = screenPoints.last,
           let firstPredicted = predictedTouchPoints.first,
           last == firstPredicted {
            screenPoints.append(contentsOf: predictedTouchPoints.dropFirst())
        } else {
            screenPoints.append(contentsOf: predictedTouchPoints)
        }

        guard !screenPoints.isEmpty else { return nil }

        // Keep the live preview cheap and bounded.
        let maxScreenPoints = 1000
        if screenPoints.count > maxScreenPoints {
            let step = max(1, screenPoints.count / maxScreenPoints)
            var downsampled: [CGPoint] = []
            downsampled.reserveCapacity(maxScreenPoints + 1)
            for i in stride(from: 0, to: screenPoints.count, by: step) {
                downsampled.append(screenPoints[i])
            }
            if let last = screenPoints.last, last != downsampled.last {
                downsampled.append(last)
            }
            screenPoints = downsampled
        }

        return screenPoints
    }

    private func buildLassoScreenPoints() -> [CGPoint]? {
        guard !lassoDrawingPoints.isEmpty else { return nil }

        var screenPoints = lassoDrawingPoints
        if let last = screenPoints.last,
           let firstPredicted = lassoPredictedPoints.first,
           last == firstPredicted {
            screenPoints.append(contentsOf: lassoPredictedPoints.dropFirst())
        } else {
            screenPoints.append(contentsOf: lassoPredictedPoints)
        }

        guard !screenPoints.isEmpty else { return nil }

        let maxScreenPoints = 1000
        if screenPoints.count > maxScreenPoints {
            let step = max(1, screenPoints.count / maxScreenPoints)
            var downsampled: [CGPoint] = []
            downsampled.reserveCapacity(maxScreenPoints + 1)
            for i in stride(from: 0, to: screenPoints.count, by: step) {
                downsampled.append(screenPoints[i])
            }
            if let last = screenPoints.last, last != downsampled.last {
                downsampled.append(last)
            }
            screenPoints = downsampled
        }

        return screenPoints
    }

    private func updateLassoPreviewFromScreenPoints(_ screenPoints: [CGPoint], close: Bool, viewSize: CGSize) {
        let worldPoints = screenPoints.map {
            screenToWorldPixels_PureDouble(
                $0,
                viewSize: viewSize,
                panOffset: panOffset,
                zoomScale: zoomScale,
                rotationAngle: rotationAngle
            )
        }

        switch lassoTarget {
        case .card(let card, let frame):
            guard let transform = transformFromActive(to: frame) else {
                updateLassoPreview(for: worldPoints, close: close)
                return
            }
            let cardPoints = worldPoints.map { activePoint in
                let framePoint = SIMD2<Double>(
                    activePoint.x * transform.scale + transform.translation.x,
                    activePoint.y * transform.scale + transform.translation.y
                )
                return framePointToCardLocal(framePoint, card: card)
            }
            let zoomInFrame = zoomScale / max(transform.scale, 1e-6)
            updateLassoPreview(for: cardPoints, close: close, card: card, frame: frame, zoom: zoomInFrame)
        default:
            updateLassoPreview(for: worldPoints, close: close)
        }
    }

    private func renderLiveStroke(view: MTKView, encoder enc: MTLRenderCommandEncoder, cameraCenterWorld: SIMD2<Double>, tempOrigin: SIMD2<Double>) {
        guard brushSettings.toolMode == .paint else { return }
        guard let target = currentDrawingTarget else { return }
        guard let screenPoints = buildLiveScreenPoints() else { return }
        guard let firstScreenPoint = screenPoints.first else { return }
        let previewColor = brushSettings.color
        enc.setCullMode(.none)

        switch target {
        case .canvas(let frame):
            let zoom: Double
            let cameraCenterInTarget: SIMD2<Double>
            if frame === activeFrame {
                zoom = max(zoomScale, 1e-6)
                cameraCenterInTarget = cameraCenterWorld
            } else if let transform = transformFromActive(to: frame) {
                zoom = max(zoomScale / transform.scale, 1e-6)
                cameraCenterInTarget = cameraCenterWorld * transform.scale + transform.translation
            } else {
                zoom = max(zoomScale, 1e-6)
                cameraCenterInTarget = cameraCenterWorld
            }
            let angle = Double(rotationAngle)
            let c = cos(angle)
            let s = sin(angle)

            let localPoints: [SIMD2<Float>] = screenPoints.map { pt in
                let dx = Double(pt.x) - Double(firstScreenPoint.x)
                let dy = Double(pt.y) - Double(firstScreenPoint.y)

                // Inverse of shader's CW matrix: [c, s; -s, c]
                let unrotatedX = dx * c + dy * s
                let unrotatedY = -dx * s + dy * c

                return SIMD2<Float>(Float(unrotatedX / zoom), Float(unrotatedY / zoom))
            }

            let segments = Stroke.buildSegments(from: localPoints, color: previewColor)
            guard !segments.isEmpty else { return }
            guard let segmentBuffer = device.makeBuffer(bytes: segments,
                                                        length: segments.count * MemoryLayout<StrokeSegmentInstance>.stride,
                                                        options: .storageModeShared) else { return }

            let dx = tempOrigin.x - cameraCenterInTarget.x
            let dy = tempOrigin.y - cameraCenterInTarget.y
            let rotatedOffsetScreen = SIMD2<Float>(
                Float((dx * c - dy * s) * zoom),
                Float((dx * s + dy * c) * zoom)
            )

            let basePixelWidth: Float
            if brushSettings.constantScreenSize {
                basePixelWidth = Float(brushSettings.size)
            } else {
                basePixelWidth = Float(brushSettings.size) * Float(zoom)
            }
            let halfPixelWidth = max(basePixelWidth * 0.5, 0.5)

            // Canvas live stroke depth: use peek for current drawing stroke
            let canvasLiveStrokeDepth = strokeDepth(for: peekStrokeDepthID())

            var transform = StrokeTransform(
                relativeOffset: .zero,
                rotatedOffsetScreen: rotatedOffsetScreen,
                zoomScale: Float(zoom),
                screenWidth: Float(view.bounds.size.width),
                screenHeight: Float(view.bounds.size.height),
                rotationAngle: rotationAngle,
                halfPixelWidth: halfPixelWidth,
                featherPx: 1.0,
                depth: canvasLiveStrokeDepth
            )

            enc.setRenderPipelineState(strokeSegmentPipelineState)
            // Use the same depth state as the final stroke will use
            enc.setDepthStencilState(brushSettings.depthWriteEnabled ? strokeDepthStateWrite : strokeDepthStateNoWrite)
            enc.setVertexBuffer(quadVertexBuffer, offset: 0, index: 0)
            enc.setVertexBytes(&transform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
            enc.setFragmentBytes(&transform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
            enc.setVertexBuffer(segmentBuffer, offset: 0, index: 2)
            enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: segments.count)

        case .card(let card, let frame):
            // Determine the camera transform in the target frame (linked list chain).
            var cameraCenterInTarget = cameraCenterWorld
            var zoomInTarget = zoomScale
            if let transform = transformFromActive(to: frame) {
                cameraCenterInTarget = cameraCenterWorld * transform.scale + transform.translation
                zoomInTarget = zoomScale / transform.scale
            }

            let liveStroke = createStrokeForCard(screenPoints: screenPoints,
                                                 card: card,
                                                 frame: frame,
                                                 viewSize: view.bounds.size,
                                                 depthID: peekStrokeDepthID(),
                                                 color: previewColor)
            guard !liveStroke.segments.isEmpty, let segmentBuffer = liveStroke.segmentBuffer else { return }

            // 1) Write stencil for the card region without affecting color output.
            let cardOffsetLocal = cardOffsetFromCameraInLocalSpace(card: card,
                                                                   cameraCenter: cameraCenterInTarget)
            let relativeOffset = SIMD2<Float>(Float(cardOffsetLocal.x), Float(cardOffsetLocal.y))
            let finalRotation = rotationAngle + card.rotation
            let cardHalfSize = SIMD2<Float>(Float(card.size.x * 0.5), Float(card.size.y * 0.5))
            var style = CardStyleUniforms(
                cardHalfSize: cardHalfSize,
                zoomScale: Float(zoomInTarget),
                cornerRadiusPx: cardCornerRadiusPx,
                shadowBlurPx: cardShadowBlurPx,
                shadowOpacity: cardShadowOpacity,
                cardOpacity: card.opacity
            )

            var cardTransform = CardTransform(
                relativeOffset: relativeOffset,
                zoomScale: Float(zoomInTarget),
                screenWidth: Float(view.bounds.size.width),
                screenHeight: Float(view.bounds.size.height),
                rotationAngle: finalRotation,
                depth: 1.0
            )

            let cardVertexBuffer = device.makeBuffer(bytes: card.localVertices,
                                                     length: card.localVertices.count * MemoryLayout<StrokeVertex>.stride,
                                                     options: .storageModeShared)

            enc.setDepthStencilState(stencilStateWrite)
            enc.setStencilReferenceValue(1)
            enc.setRenderPipelineState(cardSolidPipelineState)
            enc.setVertexBytes(&cardTransform, length: MemoryLayout<CardTransform>.stride, index: 1)
            var transparent = SIMD4<Float>(0, 0, 0, 0)
            enc.setFragmentBytes(&transparent, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
            enc.setFragmentBytes(&style, length: MemoryLayout<CardStyleUniforms>.stride, index: 2)
            enc.setVertexBuffer(cardVertexBuffer, offset: 0, index: 0)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)

            // 2) Draw the live stroke clipped to stencil.
            enc.setRenderPipelineState(strokeSegmentPipelineState)
            enc.setDepthStencilState(cardStrokeDepthStateNoWrite)
            enc.setStencilReferenceValue(1)
            enc.setVertexBuffer(quadVertexBuffer, offset: 0, index: 0)

            let totalRotation = rotationAngle + card.rotation
            let offset = SIMD2<Float>(Float(cardOffsetLocal.x), Float(cardOffsetLocal.y))

            let strokeOffset = liveStroke.origin
            let strokeRelativeOffset = offset + SIMD2<Float>(Float(strokeOffset.x), Float(strokeOffset.y))

            let basePixelWidth = Float(liveStroke.worldWidth * zoomInTarget)
            let halfPixelWidth = max(basePixelWidth * 0.5, 0.5)

            // Live stroke depth: use peekStrokeDepthID (creation order)
            let liveStrokeDepth = strokeDepth(for: peekStrokeDepthID())

            var strokeTransform = StrokeTransform(
                relativeOffset: strokeRelativeOffset,
                rotatedOffsetScreen: SIMD2<Float>(
                    Float((Double(strokeRelativeOffset.x) * cos(Double(totalRotation)) - Double(strokeRelativeOffset.y) * sin(Double(totalRotation))) * zoomInTarget),
                    Float((Double(strokeRelativeOffset.x) * sin(Double(totalRotation)) + Double(strokeRelativeOffset.y) * cos(Double(totalRotation))) * zoomInTarget)
                ),
                zoomScale: Float(zoomInTarget),
                screenWidth: Float(view.bounds.size.width),
                screenHeight: Float(view.bounds.size.height),
                rotationAngle: totalRotation,
                halfPixelWidth: halfPixelWidth,
                featherPx: 1.0,
                depth: liveStrokeDepth
            )

            enc.setVertexBytes(&strokeTransform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
            enc.setFragmentBytes(&strokeTransform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
            enc.setVertexBuffer(segmentBuffer, offset: 0, index: 2)
            enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: liveStroke.segments.count)

            // 3) Clear stencil for subsequent draws.
            enc.setRenderPipelineState(cardSolidPipelineState)
            enc.setDepthStencilState(stencilStateClear)
            enc.setStencilReferenceValue(0)
            enc.setVertexBytes(&cardTransform, length: MemoryLayout<CardTransform>.stride, index: 1)
            enc.setFragmentBytes(&transparent, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
            enc.setFragmentBytes(&style, length: MemoryLayout<CardStyleUniforms>.stride, index: 2)
            enc.setVertexBuffer(cardVertexBuffer, offset: 0, index: 0)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)

            enc.setDepthStencilState(stencilStateDefault)
        }
    }
    // Note: ICB encoding functions removed - using simple GPU-offset rendering

    private func renderLassoOverlay(view: MTKView,
                                    encoder enc: MTLRenderCommandEncoder,
                                    cameraCenterWorld: SIMD2<Double>) {
        guard lassoPreviewCard == nil else { return }
        guard let stroke = lassoPreviewStroke, lassoPreviewFrame === activeFrame else { return }
        guard !stroke.segments.isEmpty, let segmentBuffer = stroke.segmentBuffer else { return }

        enc.setRenderPipelineState(strokeSegmentPipelineState)
        enc.setDepthStencilState(strokeDepthStateNoWrite)
        enc.setCullMode(.none)
        enc.setVertexBuffer(quadVertexBuffer, offset: 0, index: 0)

        let zoom = max(zoomScale, 1e-6)
        let angle = Double(rotationAngle)
        let c = cos(angle)
        let s = sin(angle)

        let dx = stroke.origin.x - cameraCenterWorld.x
        let dy = stroke.origin.y - cameraCenterWorld.y
        let rotatedOffsetScreen = SIMD2<Float>(
            Float((dx * c - dy * s) * zoom),
            Float((dx * s + dy * c) * zoom)
        )

        let basePixelWidth = Float(stroke.worldWidth * zoom)
        let halfPixelWidth = max(basePixelWidth * 0.5, 0.5)

        var transform = StrokeTransform(
            relativeOffset: .zero,
            rotatedOffsetScreen: rotatedOffsetScreen,
            zoomScale: Float(zoom),
            screenWidth: Float(view.bounds.size.width),
            screenHeight: Float(view.bounds.size.height),
            rotationAngle: rotationAngle,
            halfPixelWidth: halfPixelWidth,
            featherPx: 1.0,
            depth: 0.0
        )

        enc.setVertexBytes(&transform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
        enc.setFragmentBytes(&transform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
        enc.setVertexBuffer(segmentBuffer, offset: 0, index: 2)
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: stroke.segments.count)
    }

    private func drawLiveEraserOnCanvasIfNeeded(frame: Frame,
                                                cameraCenterInThisFrame: SIMD2<Double>,
                                                viewSize: CGSize,
                                                currentZoom: Double,
                                                currentRotation: Float,
                                                encoder: MTLRenderCommandEncoder) {
        guard brushSettings.isMaskEraser else { return }
        guard let target = currentDrawingTarget else { return }
        guard case .canvas(let targetFrame) = target, targetFrame === frame else { return }
        guard let tempOrigin = liveStrokeOrigin else { return }
        guard let screenPoints = buildLiveScreenPoints() else { return }
        guard let firstScreenPoint = screenPoints.first else { return }

        let zoom = max(currentZoom, 1e-6)
        let angle = Double(currentRotation)
        let c = cos(angle)
        let s = sin(angle)

        let localPoints: [SIMD2<Float>] = screenPoints.map { pt in
            let dx = Double(pt.x) - Double(firstScreenPoint.x)
            let dy = Double(pt.y) - Double(firstScreenPoint.y)

            let unrotatedX = dx * c + dy * s
            let unrotatedY = -dx * s + dy * c

            return SIMD2<Float>(Float(unrotatedX / zoom), Float(unrotatedY / zoom))
        }

        let segments = Stroke.buildSegments(from: localPoints, color: SIMD4<Float>(0, 0, 0, 0))
        guard !segments.isEmpty else { return }
        guard let segmentBuffer = device.makeBuffer(bytes: segments,
                                                    length: segments.count * MemoryLayout<StrokeSegmentInstance>.stride,
                                                    options: .storageModeShared) else { return }

        let dx = tempOrigin.x - cameraCenterInThisFrame.x
        let dy = tempOrigin.y - cameraCenterInThisFrame.y
        let rotatedOffsetScreen = SIMD2<Float>(
            Float((dx * c - dy * s) * zoom),
            Float((dx * s + dy * c) * zoom)
        )

        let halfPixelWidth = max(Float(brushSettings.size) * 0.5, 0.5)
        let liveDepth = strokeDepth(for: peekStrokeDepthID())

        var transform = StrokeTransform(
            relativeOffset: .zero,
            rotatedOffsetScreen: rotatedOffsetScreen,
            zoomScale: Float(zoom),
            screenWidth: Float(viewSize.width),
            screenHeight: Float(viewSize.height),
            rotationAngle: currentRotation,
            halfPixelWidth: halfPixelWidth,
            featherPx: 1.0,
            depth: liveDepth
        )

        encoder.setRenderPipelineState(strokeSegmentPipelineState)
        encoder.setDepthStencilState(strokeDepthStateWrite)
        encoder.setVertexBuffer(quadVertexBuffer, offset: 0, index: 0)
        encoder.setVertexBytes(&transform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
        encoder.setFragmentBytes(&transform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
        encoder.setVertexBuffer(segmentBuffer, offset: 0, index: 2)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: segments.count)
    }

    ///  CONSTANT SCREEN SIZE HANDLES
    /// Draws resize handles at card corners that maintain constant screen size regardless of zoom.
    /// Rendered as rounded line segments inset from the card edges.
    ///
    /// - Parameters:
    ///   - card: The card to draw handles for
    ///   - cameraCenter: Camera position in frame coordinates
    ///   - viewSize: Screen dimensions
    ///   - zoom: Current zoom level
    ///   - rotation: Camera rotation (card rotation is baked into corner positions)
    ///   - encoder: Metal render encoder
    func drawCardHandles(card: Card,
                         cameraCenter: SIMD2<Double>,
                         viewSize: CGSize,
                         zoom: Double,
                         rotation: Float,
                         encoder: MTLRenderCommandEncoder) {

        guard zoom.isFinite, zoom > 0 else { return }

        let handleInsetPx: Float = 5.0
        let handleLengthPx: Float = 12.0
        let handleThicknessPx: Float = 3.0
        let handleColor = cardHandleColor(for: card)

        let insetWorld = handleInsetPx / Float(zoom)
        let outerCornerRadiusWorld = cardCornerRadiusPx / Float(zoom)
        let handleArcRadiusPx = max(cardCornerRadiusPx - handleInsetPx, 0.0)
        let handleArcRadiusWorld = handleArcRadiusPx / Float(zoom)
        let joinInsetWorld = handleArcRadiusWorld
        let straightLengthPx = max(handleLengthPx - handleArcRadiusPx, handleThicknessPx)

        let maxLengthX = max(0.0, Float(card.size.x) - insetWorld * 2.0 - joinInsetWorld)
        let maxLengthY = max(0.0, Float(card.size.y) - insetWorld * 2.0 - joinInsetWorld)
        let lengthWorldX = min(straightLengthPx / Float(zoom), maxLengthX)
        let lengthWorldY = min(straightLengthPx / Float(zoom), maxLengthY)

        if lengthWorldX <= 0 || lengthWorldY <= 0 { return }

        let halfW = Float(card.size.x) * 0.5
        let halfH = Float(card.size.y) * 0.5
        let xRight = halfW - insetWorld
        let yBottom = halfH - insetWorld
        let xRightStart = xRight - joinInsetWorld
        let yBottomStart = yBottom - joinInsetWorld

        struct HandleSegment {
            let p0: SIMD2<Double>
            let p1: SIMD2<Double>
        }

        let segments: [HandleSegment] = [
            // Bottom-right only (L shape)
            HandleSegment(p0: SIMD2<Double>(Double(xRightStart), Double(yBottom)),
                          p1: SIMD2<Double>(Double(xRightStart - lengthWorldX), Double(yBottom))),
            HandleSegment(p0: SIMD2<Double>(Double(xRight), Double(yBottomStart)),
                          p1: SIMD2<Double>(Double(xRight), Double(yBottomStart - lengthWorldY)))
        ]

        let angle = Double(rotation)
        let c = cos(angle)
        let s = sin(angle)

        encoder.setRenderPipelineState(strokeSegmentPipelineState)
        encoder.setDepthStencilState(stencilStateDefault)
        encoder.setVertexBuffer(quadVertexBuffer, offset: 0, index: 0)

        let drawSegment: (SIMD2<Double>, SIMD2<Double>, Float) -> Void = { p0Local, p1Local, radiusPx in
            let p0Frame = self.cardLocalToFramePoint(p0Local, card: card)
            let p1Frame = self.cardLocalToFramePoint(p1Local, card: card)
            let origin = SIMD2<Double>((p0Frame.x + p1Frame.x) * 0.5,
                                       (p0Frame.y + p1Frame.y) * 0.5)

            let localP0 = SIMD2<Float>(Float(p0Frame.x - origin.x),
                                       Float(p0Frame.y - origin.y))
            let localP1 = SIMD2<Float>(Float(p1Frame.x - origin.x),
                                       Float(p1Frame.y - origin.y))

            var segmentInstance = StrokeSegmentInstance(p0: localP0, p1: localP1, color: handleColor)
            guard let segmentBuffer = self.device.makeBuffer(bytes: &segmentInstance,
                                                        length: MemoryLayout<StrokeSegmentInstance>.stride,
                                                        options: .storageModeShared) else { return }

            let dx = origin.x - cameraCenter.x
            let dy = origin.y - cameraCenter.y
            let rotatedOffsetScreen = SIMD2<Float>(
                Float((dx * c - dy * s) * zoom),
                Float((dx * s + dy * c) * zoom)
            )

            var transform = StrokeTransform(
                relativeOffset: .zero,
                rotatedOffsetScreen: rotatedOffsetScreen,
                zoomScale: Float(zoom),
                screenWidth: Float(viewSize.width),
                screenHeight: Float(viewSize.height),
                rotationAngle: rotation,
                halfPixelWidth: radiusPx,
                featherPx: 1.0,
                depth: 0.0
            )

            encoder.setVertexBytes(&transform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
            encoder.setFragmentBytes(&transform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
            encoder.setVertexBuffer(segmentBuffer, offset: 0, index: 2)
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: 1)
        }

        for segment in segments {
            drawSegment(segment.p0, segment.p1, handleThicknessPx * 0.5)
        }

        // Only draw corner arcs if they're visually meaningful (at least 2px)
        // This prevents overlap with straight segment end caps
        if handleArcRadiusPx >= 2.0 {
            let brCenter = SIMD2<Double>(Double(halfW - outerCornerRadiusWorld),
                                         Double(halfH - outerCornerRadiusWorld))

            let arcSegments = 6
            func arcPoints(center: SIMD2<Double>, startAngle: Double, endAngle: Double) -> [SIMD2<Double>] {
                guard arcSegments > 0 else { return [] }
                var points: [SIMD2<Double>] = []
                points.reserveCapacity(arcSegments + 1)
                for i in 0...arcSegments {
                    let t = Double(i) / Double(arcSegments)
                    let angle = startAngle + (endAngle - startAngle) * t
                    let x = center.x + Double(handleArcRadiusWorld) * cos(angle)
                    let y = center.y + Double(handleArcRadiusWorld) * sin(angle)
                    points.append(SIMD2<Double>(x, y))
                }
                return points
            }

            let arcs: [[SIMD2<Double>]] = [
                arcPoints(center: brCenter, startAngle: 0.0, endAngle: 0.5 * .pi)
            ]

            for arc in arcs {
                guard arc.count >= 2 else { continue }
                for i in 0..<(arc.count - 1) {
                    drawSegment(arc[i], arc[i + 1], handleThicknessPx * 0.5)
                }
            }
        }
    }

    func drawCardLockIcon(card: Card,
                          cameraCenter: SIMD2<Double>,
                          viewSize: CGSize,
                          zoom: Double,
                          rotation: Float,
                          encoder: MTLRenderCommandEncoder) {
        guard zoom.isFinite, zoom > 0 else { return }

        let iconWidthPx: Double = 18.0
        let iconHeightPx: Double = 20.0
        let paddingPx: Double = 8.0
        let strokePx: Float = 2.0
        let shackleInsetPx: Double = 4.0
        let shackleHeightPx: Double = 7.0

        let iconWidthWorld = iconWidthPx / zoom
        let iconHeightWorld = iconHeightPx / zoom
        let paddingWorld = paddingPx / zoom

        let minWidth = iconWidthWorld + paddingWorld * 2.0
        let minHeight = iconHeightWorld + paddingWorld * 2.0
        guard card.size.x > minWidth, card.size.y > minHeight else { return }

        let halfW = card.size.x * 0.5
        let halfH = card.size.y * 0.5

        let right = halfW - paddingWorld
        let left = right - iconWidthWorld
        let top = -halfH + paddingWorld
        let bottom = top + iconHeightWorld

        let shackleInsetWorld = shackleInsetPx / zoom
        let shackleHeightWorld = shackleHeightPx / zoom
        let shackleLeft = left + shackleInsetWorld
        let shackleRight = right - shackleInsetWorld
        let shackleBottom = top + shackleHeightWorld

        struct LockSegment {
            let p0: SIMD2<Double>
            let p1: SIMD2<Double>
        }

        let segments: [LockSegment] = [
            // Shackle
            LockSegment(p0: SIMD2<Double>(shackleLeft, top),
                        p1: SIMD2<Double>(shackleRight, top)),
            LockSegment(p0: SIMD2<Double>(shackleLeft, top),
                        p1: SIMD2<Double>(shackleLeft, shackleBottom)),
            LockSegment(p0: SIMD2<Double>(shackleRight, top),
                        p1: SIMD2<Double>(shackleRight, shackleBottom)),

            // Body
            LockSegment(p0: SIMD2<Double>(left, shackleBottom),
                        p1: SIMD2<Double>(right, shackleBottom)),
            LockSegment(p0: SIMD2<Double>(left, shackleBottom),
                        p1: SIMD2<Double>(left, bottom)),
            LockSegment(p0: SIMD2<Double>(right, shackleBottom),
                        p1: SIMD2<Double>(right, bottom)),
            LockSegment(p0: SIMD2<Double>(left, bottom),
                        p1: SIMD2<Double>(right, bottom))
        ]

        let iconColor = cardHandleColor(for: card)
        let angle = Double(rotation)
        let c = cos(angle)
        let s = sin(angle)

        encoder.setRenderPipelineState(strokeSegmentPipelineState)
        encoder.setDepthStencilState(stencilStateDefault)
        encoder.setVertexBuffer(quadVertexBuffer, offset: 0, index: 0)

        let drawSegment: (SIMD2<Double>, SIMD2<Double>) -> Void = { p0Local, p1Local in
            let p0Frame = self.cardLocalToFramePoint(p0Local, card: card)
            let p1Frame = self.cardLocalToFramePoint(p1Local, card: card)
            let origin = SIMD2<Double>((p0Frame.x + p1Frame.x) * 0.5,
                                       (p0Frame.y + p1Frame.y) * 0.5)

            let localP0 = SIMD2<Float>(Float(p0Frame.x - origin.x),
                                       Float(p0Frame.y - origin.y))
            let localP1 = SIMD2<Float>(Float(p1Frame.x - origin.x),
                                       Float(p1Frame.y - origin.y))

            var segmentInstance = StrokeSegmentInstance(p0: localP0, p1: localP1, color: iconColor)
            guard let segmentBuffer = self.device.makeBuffer(bytes: &segmentInstance,
                                                             length: MemoryLayout<StrokeSegmentInstance>.stride,
                                                             options: .storageModeShared) else { return }

            let dx = origin.x - cameraCenter.x
            let dy = origin.y - cameraCenter.y
            let rotatedOffsetScreen = SIMD2<Float>(
                Float((dx * c - dy * s) * zoom),
                Float((dx * s + dy * c) * zoom)
            )

            var transform = StrokeTransform(
                relativeOffset: .zero,
                rotatedOffsetScreen: rotatedOffsetScreen,
                zoomScale: Float(zoom),
                screenWidth: Float(viewSize.width),
                screenHeight: Float(viewSize.height),
                rotationAngle: rotation,
                halfPixelWidth: strokePx * 0.5,
                featherPx: 1.0,
                depth: 0.0
            )

            encoder.setVertexBytes(&transform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
            encoder.setFragmentBytes(&transform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
            encoder.setVertexBuffer(segmentBuffer, offset: 0, index: 2)
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: 1)
        }

        for segment in segments {
            drawSegment(segment.p0, segment.p1)
        }
    }

    private func cardHandleColor(for card: Card) -> SIMD4<Float> {
        let base = card.handleBaseColor()
        let luminance = 0.299 * base.x + 0.587 * base.y + 0.114 * base.z
        if luminance > 0.5 {
            return SIMD4<Float>(0, 0, 0, 1.0)
        }
        return SIMD4<Float>(1, 1, 1, 1.0)
    }

    func draw(in view: MTKView) {
        resizeOffscreenTexturesIfNeeded(drawableSize: view.drawableSize)
        guard let offscreenColor = offscreenColorTexture,
              let offscreenDepthStencil = offscreenDepthStencilTexture else { return }

        guard let mainCommandBuffer = commandQueue.makeCommandBuffer(),
              let drawable = view.currentDrawable,
              let drawableRPD = view.currentRenderPassDescriptor else { return }

        // PASS 1: Render the full scene into an offscreen texture (no MSAA, hard stroke coverage).
        let sceneRPD = MTLRenderPassDescriptor()
        sceneRPD.colorAttachments[0].texture = offscreenColor
        sceneRPD.colorAttachments[0].loadAction = .clear
        sceneRPD.colorAttachments[0].storeAction = .store
        sceneRPD.colorAttachments[0].clearColor = view.clearColor

        sceneRPD.depthAttachment.texture = offscreenDepthStencil
        sceneRPD.depthAttachment.loadAction = .clear
        sceneRPD.depthAttachment.storeAction = .dontCare
        sceneRPD.depthAttachment.clearDepth = view.clearDepth

        sceneRPD.stencilAttachment.texture = offscreenDepthStencil
        sceneRPD.stencilAttachment.loadAction = .clear
        sceneRPD.stencilAttachment.storeAction = .dontCare
        sceneRPD.stencilAttachment.clearStencil = view.clearStencil

        guard let sceneEnc = mainCommandBuffer.makeRenderCommandEncoder(descriptor: sceneRPD) else { return }

        let cameraCenterWorld = calculateCameraCenterWorld(viewSize: view.bounds.size)

        sceneEnc.setRenderPipelineState(strokeSegmentPipelineState)
        sceneEnc.setCullMode(.none)

        renderFrame(activeFrame,
                    cameraCenterInThisFrame: cameraCenterWorld,
                    viewSize: view.bounds.size,
                    currentZoom: zoomScale,
                    currentRotation: rotationAngle,
                    encoder: sceneEnc,
                    excludedChild: nil)

        // Live stroke rendering (same logic), reusing the scene encoder.
        if !brushSettings.isLasso,
           (currentTouchPoints.count + predictedTouchPoints.count) >= 2,
           let tempOrigin = liveStrokeOrigin {
            renderLiveStroke(view: view, encoder: sceneEnc, cameraCenterWorld: cameraCenterWorld, tempOrigin: tempOrigin)
        }

        renderLassoOverlay(view: view, encoder: sceneEnc, cameraCenterWorld: cameraCenterWorld)

        sceneEnc.endEncoding()

        // PASS 2: FXAA fullscreen pass into the drawable.
        guard let postEnc = mainCommandBuffer.makeRenderCommandEncoder(descriptor: drawableRPD) else { return }
        postEnc.setRenderPipelineState(postProcessPipelineState)
        postEnc.setDepthStencilState(stencilStateDefault)
        postEnc.setCullMode(.none)

        postEnc.setFragmentTexture(offscreenColor, index: 0)
        postEnc.setFragmentSamplerState(samplerState, index: 0)
        var invResolution = SIMD2<Float>(1.0 / Float(offscreenTextureWidth),
                                         1.0 / Float(offscreenTextureHeight))
        postEnc.setFragmentBytes(&invResolution, length: MemoryLayout<SIMD2<Float>>.stride, index: 0)
        postEnc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        postEnc.endEncoding()

        mainCommandBuffer.present(drawable)
        mainCommandBuffer.commit()

        updateDebugHUD(view: view)
    }
    func updateDebugHUD(view: MTKView) {
        // Access debugLabel through the stored metalView reference
        guard let mtkView = metalView else { return }

        // Find the debug label subview
        guard let debugLabel = mtkView.subviews.compactMap({ $0 as? UILabel }).first else { return }

        // Get stored depth from the frame
        let depth = activeFrame.depthFromRoot

        // Format zoom scale nicely
        let zoomText: String
        if zoomScale >= 1000.0 {
            zoomText = String(format: "%.1fk×", zoomScale / 1000.0)
        } else if zoomScale >= 1.0 {
            zoomText = String(format: "%.1f×", zoomScale)
        } else {
            zoomText = String(format: "%.3f×", zoomScale)
        }

        // Calculate effective zoom (depth multiplier)
        // pow(1000, -1) = 0.001, so negative depths work correctly
        let effectiveZoom = pow(1000.0, Double(depth)) * zoomScale
        let effectiveText: String
        if effectiveZoom >= 1e12 {
            let exponent = Int(log10(effectiveZoom))
            effectiveText = String(format: "10^%d", exponent)
        } else if effectiveZoom >= 1e9 {
            effectiveText = String(format: "%.1fB×", effectiveZoom / 1e9)
        } else if effectiveZoom >= 1e6 {
            effectiveText = String(format: "%.1fM×", effectiveZoom / 1e6)
        } else if effectiveZoom >= 1e3 {
            effectiveText = String(format: "%.1fk×", effectiveZoom / 1e3)
        } else if effectiveZoom >= 1.0 {
            effectiveText = String(format: "%.1f×", effectiveZoom)
        } else if effectiveZoom >= 0.001 {
            effectiveText = String(format: "%.3f×", effectiveZoom)
        } else {
            let exponent = Int(log10(effectiveZoom))
            effectiveText = String(format: "10^%d", exponent)
        }

        // Calculate camera position in current frame
        let cameraPos = calculateCameraCenterWorld(viewSize: view.bounds.size)
        let cameraPosText = String(format: "(%.1f, %.1f)", cameraPos.x, cameraPos.y)

        // Update label on main thread
        DispatchQueue.main.async {
            debugLabel.text = """
            Depth: \(depth) | Zoom: \(zoomText)
            Effective: \(effectiveText)
            Strokes: \(self.activeFrame.strokes.count)
            Camera: \(cameraPosText)
            Vertices: \(self.debugDrawnVerticesThisFrame) | Draws: \(self.debugDrawnNodesThisFrame)
            """
        }
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        resizeOffscreenTexturesIfNeeded(drawableSize: size)
    }

    private func resizeOffscreenTexturesIfNeeded(drawableSize: CGSize) {
        let width = max(Int(drawableSize.width.rounded(.down)), 1)
        let height = max(Int(drawableSize.height.rounded(.down)), 1)
        guard width != offscreenTextureWidth ||
              height != offscreenTextureHeight ||
              offscreenColorTexture == nil ||
              offscreenDepthStencilTexture == nil else { return }

        offscreenTextureWidth = width
        offscreenTextureHeight = height

        let colorDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        colorDesc.usage = [.renderTarget, .shaderRead]
        colorDesc.storageMode = .private
        colorDesc.sampleCount = 1
        offscreenColorTexture = device.makeTexture(descriptor: colorDesc)

        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float_stencil8,
            width: width,
            height: height,
            mipmapped: false
        )
        depthDesc.usage = [.renderTarget]
        depthDesc.storageMode = .private
        depthDesc.sampleCount = 1
        offscreenDepthStencilTexture = device.makeTexture(descriptor: depthDesc)
    }

    private func makeQuadVertexDescriptor() -> MTLVertexDescriptor {
        let vd = MTLVertexDescriptor()
        vd.attributes[0].format = .float2
        vd.attributes[0].offset = 0
        vd.attributes[0].bufferIndex = 0
        vd.layouts[0].stride = MemoryLayout<SIMD2<Float>>.stride
        vd.layouts[0].stepFunction = .perVertex
        return vd
    }

    func makePipeLine() {
        let library = device.makeDefaultLibrary()!
        // Single-sample render target (FXAA handles anti-aliasing as a post-process).
        // Must match `MTKView.sampleCount`.
        let viewSampleCount = 1
        let depthStencilFormat: MTLPixelFormat = .depth32Float_stencil8

        let quadVertexDesc = makeQuadVertexDescriptor()

        // Stroke Pipeline (Instanced SDF segments)
        let segDesc = MTLRenderPipelineDescriptor()
        segDesc.vertexFunction   = library.makeFunction(name: "vertex_segment_sdf")
        segDesc.fragmentFunction = library.makeFunction(name: "fragment_segment_sdf")
        segDesc.colorAttachments[0].pixelFormat = .bgra8Unorm
        segDesc.sampleCount = viewSampleCount

        let segAttachment = segDesc.colorAttachments[0]!
        segAttachment.isBlendingEnabled = true
        segAttachment.rgbBlendOperation = .add
        segAttachment.alphaBlendOperation = .add
        segAttachment.sourceRGBBlendFactor = .sourceAlpha
        segAttachment.sourceAlphaBlendFactor = .sourceAlpha
        segAttachment.destinationRGBBlendFactor = .oneMinusSourceAlpha
        segAttachment.destinationAlphaBlendFactor = .oneMinusSourceAlpha

        segDesc.vertexDescriptor = quadVertexDesc
        segDesc.depthAttachmentPixelFormat = depthStencilFormat
        segDesc.stencilAttachmentPixelFormat = depthStencilFormat
        do {
            strokeSegmentPipelineState = try device.makeRenderPipelineState(descriptor: segDesc)
        } catch {
            fatalError("Failed to create strokeSegmentPipelineState: \(error)")
        }

        // Setup vertex descriptor for StrokeVertex structure (shared by both card pipelines)
        let vertexDesc = MTLVertexDescriptor()
        vertexDesc.attributes[0].format = .float2
        vertexDesc.attributes[0].offset = 0
        vertexDesc.attributes[0].bufferIndex = 0
        vertexDesc.attributes[1].format = .float2
        vertexDesc.attributes[1].offset = MemoryLayout<SIMD2<Float>>.stride
        vertexDesc.attributes[1].bufferIndex = 0
        vertexDesc.layouts[0].stride = MemoryLayout<StrokeVertex>.stride
        vertexDesc.layouts[0].stepFunction = .perVertex

        // Textured Card Pipeline (for images, PDFs)
        let cardDesc = MTLRenderPipelineDescriptor()
        cardDesc.vertexFunction   = library.makeFunction(name: "vertex_card")
        cardDesc.fragmentFunction = library.makeFunction(name: "fragment_card_texture")
        cardDesc.colorAttachments[0].pixelFormat = .bgra8Unorm
        cardDesc.sampleCount = viewSampleCount
        cardDesc.colorAttachments[0].isBlendingEnabled = true
        cardDesc.colorAttachments[0].rgbBlendOperation = .add
        cardDesc.colorAttachments[0].alphaBlendOperation = .add
        cardDesc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        cardDesc.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        cardDesc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        cardDesc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        cardDesc.vertexDescriptor = vertexDesc
        cardDesc.depthAttachmentPixelFormat = depthStencilFormat
        cardDesc.stencilAttachmentPixelFormat = depthStencilFormat // Required for stencil buffer
        do {
            cardPipelineState = try device.makeRenderPipelineState(descriptor: cardDesc)
        } catch {
            fatalError("Failed to create cardPipelineState: \(error)")
        }

        // Solid Color Card Pipeline (for placeholders, backgrounds)
        let cardSolidDesc = MTLRenderPipelineDescriptor()
        cardSolidDesc.vertexFunction   = library.makeFunction(name: "vertex_card")
        cardSolidDesc.fragmentFunction = library.makeFunction(name: "fragment_card_solid")
        cardSolidDesc.colorAttachments[0].pixelFormat = .bgra8Unorm
        cardSolidDesc.sampleCount = viewSampleCount
        cardSolidDesc.colorAttachments[0].isBlendingEnabled = true
        cardSolidDesc.colorAttachments[0].rgbBlendOperation = .add
        cardSolidDesc.colorAttachments[0].alphaBlendOperation = .add
        cardSolidDesc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        cardSolidDesc.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        cardSolidDesc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        cardSolidDesc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        cardSolidDesc.vertexDescriptor = vertexDesc
        cardSolidDesc.depthAttachmentPixelFormat = depthStencilFormat
        cardSolidDesc.stencilAttachmentPixelFormat = depthStencilFormat // Required for stencil buffer

        do {
            cardSolidPipelineState = try device.makeRenderPipelineState(descriptor: cardSolidDesc)
        } catch {
            fatalError("Failed to create solid card pipeline: \(error)")
        }

        // Lined Card Pipeline (Procedural horizontal lines)
        let linedDesc = MTLRenderPipelineDescriptor()
        linedDesc.vertexFunction = library.makeFunction(name: "vertex_card")
        linedDesc.fragmentFunction = library.makeFunction(name: "fragment_card_lined")
        linedDesc.colorAttachments[0].pixelFormat = .bgra8Unorm
        linedDesc.sampleCount = viewSampleCount
        linedDesc.colorAttachments[0].isBlendingEnabled = true
        linedDesc.colorAttachments[0].rgbBlendOperation = .add
        linedDesc.colorAttachments[0].alphaBlendOperation = .add
        linedDesc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        linedDesc.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        linedDesc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        linedDesc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        linedDesc.vertexDescriptor = vertexDesc
        linedDesc.depthAttachmentPixelFormat = depthStencilFormat
        linedDesc.stencilAttachmentPixelFormat = depthStencilFormat

        do {
            cardLinedPipelineState = try device.makeRenderPipelineState(descriptor: linedDesc)
        } catch {
            fatalError("Failed to create lined card pipeline: \(error)")
        }

        // Grid Card Pipeline (Procedural horizontal and vertical lines)
        let gridDesc = MTLRenderPipelineDescriptor()
        gridDesc.vertexFunction = library.makeFunction(name: "vertex_card")
        gridDesc.fragmentFunction = library.makeFunction(name: "fragment_card_grid")
        gridDesc.colorAttachments[0].pixelFormat = .bgra8Unorm
        gridDesc.sampleCount = viewSampleCount
        gridDesc.colorAttachments[0].isBlendingEnabled = true
        gridDesc.colorAttachments[0].rgbBlendOperation = .add
        gridDesc.colorAttachments[0].alphaBlendOperation = .add
        gridDesc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        gridDesc.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        gridDesc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        gridDesc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        gridDesc.vertexDescriptor = vertexDesc
        gridDesc.depthAttachmentPixelFormat = depthStencilFormat
        gridDesc.stencilAttachmentPixelFormat = depthStencilFormat

        do {
            cardGridPipelineState = try device.makeRenderPipelineState(descriptor: gridDesc)
        } catch {
            fatalError("Failed to create grid card pipeline: \(error)")
        }

        // Card Shadow Pipeline
        let shadowDesc = MTLRenderPipelineDescriptor()
        shadowDesc.vertexFunction = library.makeFunction(name: "vertex_card")
        shadowDesc.fragmentFunction = library.makeFunction(name: "fragment_card_shadow")
        shadowDesc.colorAttachments[0].pixelFormat = .bgra8Unorm
        shadowDesc.sampleCount = viewSampleCount
        shadowDesc.colorAttachments[0].isBlendingEnabled = true
        shadowDesc.colorAttachments[0].rgbBlendOperation = .add
        shadowDesc.colorAttachments[0].alphaBlendOperation = .add
        shadowDesc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        shadowDesc.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        shadowDesc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        shadowDesc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        shadowDesc.vertexDescriptor = vertexDesc
        shadowDesc.depthAttachmentPixelFormat = depthStencilFormat
        shadowDesc.stencilAttachmentPixelFormat = depthStencilFormat

        do {
            cardShadowPipelineState = try device.makeRenderPipelineState(descriptor: shadowDesc)
        } catch {
            fatalError("Failed to create card shadow pipeline: \(error)")
        }

        // Post-process Pipeline (FXAA fullscreen pass)
        let fxaaDesc = MTLRenderPipelineDescriptor()
        fxaaDesc.vertexFunction = library.makeFunction(name: "vertex_fullscreen_triangle")
        fxaaDesc.fragmentFunction = library.makeFunction(name: "fragment_fxaa")
        fxaaDesc.colorAttachments[0].pixelFormat = .bgra8Unorm
        fxaaDesc.sampleCount = viewSampleCount
        fxaaDesc.depthAttachmentPixelFormat = depthStencilFormat
        fxaaDesc.stencilAttachmentPixelFormat = depthStencilFormat

        do {
            postProcessPipelineState = try device.makeRenderPipelineState(descriptor: fxaaDesc)
        } catch {
            fatalError("Failed to create postProcessPipelineState: \(error)")
        }

        // Create Sampler for card textures
        let samplerDesc = MTLSamplerDescriptor()
        samplerDesc.minFilter = .linear
        samplerDesc.magFilter = .linear
        samplerDesc.mipFilter = .linear
        samplerDesc.sAddressMode = .clampToEdge
        samplerDesc.tAddressMode = .clampToEdge
        samplerState = device.makeSamplerState(descriptor: samplerDesc)

        // Initialize stencil states for card clipping
        makeDepthStencilStates()
    }

    func makeDepthStencilStates() {
        let desc = MTLDepthStencilDescriptor()
        desc.depthCompareFunction = .always
        desc.isDepthWriteEnabled = false

        // 0. DEFAULT State (Passthrough - no stencil testing or writing)
        // "Always pass, keep everything as-is"
        let stencilDefault = MTLStencilDescriptor()
        stencilDefault.stencilCompareFunction = .always
        stencilDefault.stencilFailureOperation = .keep
        stencilDefault.depthFailureOperation = .keep
        stencilDefault.depthStencilPassOperation = .keep
        desc.frontFaceStencil = stencilDefault
        desc.backFaceStencil = stencilDefault
        stencilStateDefault = device.makeDepthStencilState(descriptor: desc)

        // 1. WRITE State (Used when drawing the Card Background)
        // "Always pass, replace stencil with the reference (1), and reset depth to far"
        let stencilWrite = MTLStencilDescriptor()
        stencilWrite.stencilCompareFunction = .always
        stencilWrite.stencilFailureOperation = .keep
        stencilWrite.depthFailureOperation = .keep
        stencilWrite.depthStencilPassOperation = .replace
        let stencilWriteDesc = MTLDepthStencilDescriptor()
        stencilWriteDesc.depthCompareFunction = .always
        stencilWriteDesc.isDepthWriteEnabled = true
        stencilWriteDesc.frontFaceStencil = stencilWrite
        stencilWriteDesc.backFaceStencil = stencilWrite
        stencilStateWrite = device.makeDepthStencilState(descriptor: stencilWriteDesc)

        // 2. READ State (Stencil-only, no depth testing)
        // "Only pass if the stencil value equals the reference (1)"
        let stencilRead = MTLStencilDescriptor()
        stencilRead.stencilCompareFunction = .equal
        stencilRead.stencilFailureOperation = .keep
        stencilRead.depthFailureOperation = .keep
        stencilRead.depthStencilPassOperation = .keep
        desc.frontFaceStencil = stencilRead
        desc.backFaceStencil = stencilRead
        stencilStateRead = device.makeDepthStencilState(descriptor: desc)

        // 3. CLEAR State (Used to clean up after a card)
        // "Always pass, and replace stencil with 0"
        let stencilClear = MTLStencilDescriptor()
        stencilClear.stencilCompareFunction = .always
        stencilClear.depthStencilPassOperation = .zero // Reset to 0
        desc.frontFaceStencil = stencilClear
        desc.backFaceStencil = stencilClear
        stencilStateClear = device.makeDepthStencilState(descriptor: desc)

        // Depth-only states for strokes (no stencil).
        let strokeWriteDesc = MTLDepthStencilDescriptor()
        strokeWriteDesc.depthCompareFunction = .less
        strokeWriteDesc.isDepthWriteEnabled = true
        strokeDepthStateWrite = device.makeDepthStencilState(descriptor: strokeWriteDesc)

        let strokeNoWriteDesc = MTLDepthStencilDescriptor()
        strokeNoWriteDesc.depthCompareFunction = .less
        strokeNoWriteDesc.isDepthWriteEnabled = false
        strokeDepthStateNoWrite = device.makeDepthStencilState(descriptor: strokeNoWriteDesc)

        let cardStencilRead = MTLStencilDescriptor()
        cardStencilRead.stencilCompareFunction = .equal
        cardStencilRead.stencilFailureOperation = .keep
        cardStencilRead.depthFailureOperation = .keep
        cardStencilRead.depthStencilPassOperation = .keep

        let cardStrokeWriteDesc = MTLDepthStencilDescriptor()
        cardStrokeWriteDesc.depthCompareFunction = .less
        cardStrokeWriteDesc.isDepthWriteEnabled = true
        cardStrokeWriteDesc.frontFaceStencil = cardStencilRead
        cardStrokeWriteDesc.backFaceStencil = cardStencilRead
        cardStrokeDepthStateWrite = device.makeDepthStencilState(descriptor: cardStrokeWriteDesc)

        let cardStrokeNoWriteDesc = MTLDepthStencilDescriptor()
        cardStrokeNoWriteDesc.depthCompareFunction = .less
        cardStrokeNoWriteDesc.isDepthWriteEnabled = false
        cardStrokeNoWriteDesc.frontFaceStencil = cardStencilRead
        cardStrokeNoWriteDesc.backFaceStencil = cardStencilRead
        cardStrokeDepthStateNoWrite = device.makeDepthStencilState(descriptor: cardStrokeNoWriteDesc)
	    }

    func makeVertexBuffer() {
        var positions: [SIMD2<Float>] = [
            SIMD2<Float>(-0.8,  0.5),
            SIMD2<Float>(-0.3, -0.5),
            SIMD2<Float>(-0.8, -0.5),
            SIMD2<Float>(-0.3, -0.5),
            SIMD2<Float>(-0.3,  0.5),
            SIMD2<Float>(-0.8,  0.5),
        ]
        vertexBuffer = device.makeBuffer(bytes: &positions,
                                         length: positions.count * MemoryLayout<SIMD2<Float>>.stride,
                                         options: [])
    }

    func updateVertexBuffer(with vertices: [SIMD2<Float>]) {
        guard !vertices.isEmpty else { return }
        let bufferSize = vertices.count * MemoryLayout<SIMD2<Float>>.stride
        vertexBuffer = device.makeBuffer(bytes: vertices, length: bufferSize, options: .storageModeShared)
    }

    func makeQuadVertexBuffer() {
        let quadVertices: [QuadVertex] = [
            QuadVertex(corner: SIMD2<Float>(0, 0)),
            QuadVertex(corner: SIMD2<Float>(1, 0)),
            QuadVertex(corner: SIMD2<Float>(0, 1)),
            QuadVertex(corner: SIMD2<Float>(1, 1)),
        ]

        quadVertexBuffer = device.makeBuffer(bytes: quadVertices,
                                             length: quadVertices.count * MemoryLayout<QuadVertex>.stride,
                                             options: .storageModeShared)
    }

    // MARK: - Camera Center Calculation

    /// Calculate the camera center in world coordinates using Double precision.
    /// This is the inverse of the pan/zoom/rotate transform applied to strokes.
    func calculateCameraCenterWorld(viewSize: CGSize) -> SIMD2<Double> {
        // The center of the screen in screen coordinates
        let screenCenter = CGPoint(x: viewSize.width / 2.0, y: viewSize.height / 2.0)

        //  USE PURE DOUBLE HELPER
        // Pass Double panOffset and zoomScale directly without casting to Float
        // This prevents precision loss at extreme zoom levels (1,000,000x+)
        return screenToWorldPixels_PureDouble(screenCenter,
                                              viewSize: viewSize,
                                              panOffset: panOffset,       // Now passing SIMD2<Double>
                                              zoomScale: zoomScale,       // Now passing Double
                                              rotationAngle: rotationAngle)
    }

    private func cameraCenterInFrame(_ frame: Frame,
                                     cameraCenterActive: SIMD2<Double>) -> SIMD2<Double> {
        guard frame !== activeFrame, let transform = transformFromActive(to: frame) else {
            return cameraCenterActive
        }
        return cameraCenterActive * transform.scale + transform.translation
    }

    private func cameraCenterAndZoom(in frame: Frame,
                                     cameraCenterActive: SIMD2<Double>) -> (SIMD2<Double>, Double) {
        guard frame !== activeFrame, let transform = transformFromActive(to: frame) else {
            return (cameraCenterActive, max(zoomScale, 1e-6))
        }
        let cameraCenter = cameraCenterActive * transform.scale + transform.translation
        let effectiveZoom = max(zoomScale / transform.scale, 1e-6)
        return (cameraCenter, effectiveZoom)
    }

    private func randomStrokePoints(center: SIMD2<Double>, maxOffset: Double) -> [SIMD2<Double>] {
        let pointCount = Int.random(in: 2...6)
        var points: [SIMD2<Double>] = []
        points.reserveCapacity(pointCount)

        var x = center.x + Double.random(in: -maxOffset...maxOffset)
        var y = center.y + Double.random(in: -maxOffset...maxOffset)
        points.append(SIMD2<Double>(x, y))

        let stepRange = maxOffset * 0.1
        for _ in 1..<pointCount {
            x += Double.random(in: -stepRange...stepRange)
            y += Double.random(in: -stepRange...stepRange)
            x = min(center.x + maxOffset, max(center.x - maxOffset, x))
            y = min(center.y + maxOffset, max(center.y - maxOffset, y))
            points.append(SIMD2<Double>(x, y))
        }

        return points
    }

    // MARK: - Stroke Simplification

    /// Simplify a stroke in screen space by removing points that are too close
    /// or nearly collinear with neighbors.
    /// This prevents vertex explosion and reduces overdraw without noticeable quality loss.
    ///
    /// - Parameters:
    ///   - points: Input stroke points in screen space
    ///   - minScreenDist: Minimum distance between points in screen pixels (default: 1.5)
    ///   - minAngleDeg: Minimum angle change to keep a point in degrees (default: 5.0)
    /// - Returns: Simplified array of points
    func simplifyStroke(
        _ points: [CGPoint],
        minScreenDist: CGFloat = 1.5,
        minAngleDeg: CGFloat = 5.0
    ) -> [CGPoint] {
        guard points.count > 2 else { return points }

        var result: [CGPoint] = [points[0]]

        for i in 1..<(points.count - 1) {
            let prev = result.last!
            let cur  = points[i]
            let next = points[i + 1]

            // Distance filter
            let dx = cur.x - prev.x
            let dy = cur.y - prev.y
            let dist2 = dx*dx + dy*dy
            if dist2 < minScreenDist * minScreenDist {
                continue
            }

            // Angle filter
            let v1 = CGPoint(x: cur.x - prev.x, y: cur.y - prev.y)
            let v2 = CGPoint(x: next.x - cur.x, y: next.y - cur.y)
            let len1 = hypot(v1.x, v1.y)
            let len2 = hypot(v2.x, v2.y)
            if len1 > 0, len2 > 0 {
                let dot = (v1.x * v2.x + v1.y * v2.y) / (len1 * len2)
                let clampedDot = max(-1.0, min(1.0, dot))
                let angle = acos(clampedDot) * 180.0 / .pi

                if angle < minAngleDeg {
                    // Almost straight line – skip
                    continue
                }
            }

            result.append(cur)
        }

        if let last = points.last, last != result.last {
            result.append(last)
        }

        return result
    }

    // MARK: - Lasso Selection

    func clearLassoSelection() {
        lassoSelection = nil
        lassoPreviewStroke = nil
        lassoPreviewFrame = nil
        lassoTransformState = nil
        lassoTarget = nil
        lassoPreviewCard = nil
        lassoPreviewCardFrame = nil
    }

    func handleLassoTap(screenPoint: CGPoint, viewSize: CGSize) -> Bool {
        guard let selection = lassoSelection else { return false }
        let pointWorld = screenToWorldPixels_PureDouble(
            screenPoint,
            viewSize: viewSize,
            panOffset: panOffset,
            zoomScale: zoomScale,
            rotationAngle: rotationAngle
        )

        if !pointInPolygon(pointWorld, polygon: selection.points) {
            clearLassoSelection()
        }
        return true
    }

    func lassoContains(screenPoint: CGPoint, viewSize: CGSize) -> Bool {
        guard let selection = lassoSelection else { return false }
        let pointWorld = screenToWorldPixels_PureDouble(
            screenPoint,
            viewSize: viewSize,
            panOffset: panOffset,
            zoomScale: zoomScale,
            rotationAngle: rotationAngle
        )
        return pointInPolygon(pointWorld, polygon: selection.points)
    }

    func translateLassoSelection(by delta: SIMD2<Double>) {
        guard var selection = lassoSelection else { return }

        for frameSelection in selection.frames {
            guard let transform = transformFromActive(to: frameSelection.frame) else { continue }
            let deltaFrame = SIMD2<Double>(delta.x * transform.scale, delta.y * transform.scale)

            for (index, stroke) in frameSelection.frame.strokes.enumerated() {
                guard frameSelection.strokeIDs.contains(stroke.id) else { continue }
                let newOrigin = SIMD2<Double>(stroke.origin.x + deltaFrame.x, stroke.origin.y + deltaFrame.y)
                let newStroke = Stroke(
                    id: stroke.id,
                    origin: newOrigin,
                    worldWidth: stroke.worldWidth,
                    color: stroke.color,
                    zoomEffectiveAtCreation: stroke.zoomEffectiveAtCreation,
                    segments: stroke.segments,
                    localBounds: stroke.localBounds,
                    segmentBounds: stroke.segmentBounds,
                    device: device,
                    depthID: stroke.depthID,
                    depthWriteEnabled: stroke.depthWriteEnabled
                )
                frameSelection.frame.strokes[index] = newStroke
            }
        }

        for cardSelection in selection.cards {
            if cardSelection.card.isLocked { continue }
            guard let transform = transformFromActive(to: cardSelection.frame) else { continue }
            let deltaFrame = SIMD2<Double>(delta.x * transform.scale, delta.y * transform.scale)
            cardSelection.card.origin.x += deltaFrame.x
            cardSelection.card.origin.y += deltaFrame.y
        }

        for cardStrokeSelection in selection.cardStrokes {
            if cardStrokeSelection.card.isLocked { continue }
            guard let transform = transformFromActive(to: cardStrokeSelection.frame) else { continue }
            let deltaFrame = SIMD2<Double>(delta.x * transform.scale, delta.y * transform.scale)
            let c = cos(Double(-cardStrokeSelection.card.rotation))
            let s = sin(Double(-cardStrokeSelection.card.rotation))
            let localDx = deltaFrame.x * c - deltaFrame.y * s
            let localDy = deltaFrame.x * s + deltaFrame.y * c

            for (index, stroke) in cardStrokeSelection.card.strokes.enumerated() {
                guard cardStrokeSelection.strokeIDs.contains(stroke.id) else { continue }
                let newOrigin = SIMD2<Double>(stroke.origin.x + localDx, stroke.origin.y + localDy)
                let newStroke = Stroke(
                    id: stroke.id,
                    origin: newOrigin,
                    worldWidth: stroke.worldWidth,
                    color: stroke.color,
                    zoomEffectiveAtCreation: stroke.zoomEffectiveAtCreation,
                    segments: stroke.segments,
                    localBounds: stroke.localBounds,
                    segmentBounds: stroke.segmentBounds,
                    device: device,
                    depthID: stroke.depthID,
                    depthWriteEnabled: stroke.depthWriteEnabled
                )
                cardStrokeSelection.card.strokes[index] = newStroke
            }
        }

        selection.points = selection.points.map { SIMD2<Double>($0.x + delta.x, $0.y + delta.y) }
        selection.bounds = selection.bounds.offsetBy(dx: delta.x, dy: delta.y)
        selection.center = SIMD2<Double>(selection.center.x + delta.x, selection.center.y + delta.y)
        lassoSelection = selection
        updateLassoPreview(for: selection)
    }

    func beginLassoTransformIfNeeded() {
        guard lassoTransformState == nil,
              let selection = lassoSelection else { return }

        var snapshots: [StrokeSnapshot] = []
        var cardSnapshots: [CardSnapshot] = []
        var cardStrokeSnapshots: [CardStrokeSnapshot] = []

        for frameSelection in selection.frames {
            guard let transform = transformFromActive(to: frameSelection.frame) else { continue }
            let invScale = transform.scale != 0 ? (1.0 / transform.scale) : 1.0

            for (index, stroke) in frameSelection.frame.strokes.enumerated() {
                guard frameSelection.strokeIDs.contains(stroke.id) else { continue }
                let worldPointsFrame = stroke.rawPoints.map {
                    SIMD2<Double>(
                        stroke.origin.x + Double($0.x),
                        stroke.origin.y + Double($0.y)
                    )
                }
                let worldPointsActive = worldPointsFrame.map {
                    SIMD2<Double>(
                        ($0.x - transform.translation.x) * invScale,
                        ($0.y - transform.translation.y) * invScale
                    )
                }

                snapshots.append(
                    StrokeSnapshot(
                        id: stroke.id,
                        frame: frameSelection.frame,
                        index: index,
                        activePoints: worldPointsActive,
                        color: stroke.color,
                        worldWidth: stroke.worldWidth,
                        zoomEffectiveAtCreation: stroke.zoomEffectiveAtCreation,
                        depthID: stroke.depthID,
                        depthWriteEnabled: stroke.depthWriteEnabled,
                        frameScale: transform.scale,
                        frameTranslation: transform.translation
                    )
                )
            }
        }

        for cardSelection in selection.cards {
            if cardSelection.card.isLocked { continue }
            guard let transform = transformFromActive(to: cardSelection.frame) else { continue }
            let invScale = transform.scale != 0 ? (1.0 / transform.scale) : 1.0
            let originActive = SIMD2<Double>(
                (cardSelection.card.origin.x - transform.translation.x) * invScale,
                (cardSelection.card.origin.y - transform.translation.y) * invScale
            )
            cardSnapshots.append(
                CardSnapshot(
                    card: cardSelection.card,
                    frameScale: transform.scale,
                    frameTranslation: transform.translation,
                    originActive: originActive,
                    size: cardSelection.card.size,
                    rotation: cardSelection.card.rotation
                )
            )
        }

        for cardStrokeSelection in selection.cardStrokes {
            if cardStrokeSelection.card.isLocked { continue }
            guard let transform = transformFromActive(to: cardStrokeSelection.frame) else { continue }
            let invScale = transform.scale != 0 ? (1.0 / transform.scale) : 1.0
            let cardOrigin = cardStrokeSelection.card.origin
            let cardRotation = Double(cardStrokeSelection.card.rotation)
            let c = cos(cardRotation)
            let s = sin(cardRotation)

            for (index, stroke) in cardStrokeSelection.card.strokes.enumerated() {
                guard cardStrokeSelection.strokeIDs.contains(stroke.id) else { continue }

                let localPoints = stroke.rawPoints.map {
                    SIMD2<Double>(
                        stroke.origin.x + Double($0.x),
                        stroke.origin.y + Double($0.y)
                    )
                }

                let framePoints = localPoints.map {
                    let rotX = $0.x * c - $0.y * s
                    let rotY = $0.x * s + $0.y * c
                    return SIMD2<Double>(cardOrigin.x + rotX, cardOrigin.y + rotY)
                }

                let activePoints = framePoints.map {
                    SIMD2<Double>(
                        ($0.x - transform.translation.x) * invScale,
                        ($0.y - transform.translation.y) * invScale
                    )
                }

                cardStrokeSnapshots.append(
                    CardStrokeSnapshot(
                        card: cardStrokeSelection.card,
                        frame: cardStrokeSelection.frame,
                        index: index,
                        activePoints: activePoints,
                        color: stroke.color,
                        worldWidth: stroke.worldWidth,
                        zoomEffectiveAtCreation: stroke.zoomEffectiveAtCreation,
                        depthID: stroke.depthID,
                        depthWriteEnabled: stroke.depthWriteEnabled,
                        frameScale: transform.scale,
                        frameTranslation: transform.translation,
                        cardOrigin: cardOrigin,
                        cardRotation: cardRotation
                    )
                )
            }
        }

        lassoTransformState = LassoTransformState(
            basePoints: selection.points,
            baseCenter: selection.center,
            baseStrokes: snapshots,
            baseCards: cardSnapshots,
            baseCardStrokes: cardStrokeSnapshots,
            currentScale: 1.0,
            currentRotation: 0.0
        )
    }

    func updateLassoTransformScale(delta: Double) {
        guard var state = lassoTransformState else { return }
        guard delta.isFinite, delta > 0 else { return }
        state.currentScale *= delta
        lassoTransformState = state
        applyLassoTransform(state)
    }

    func updateLassoTransformRotation(delta: Double) {
        guard var state = lassoTransformState else { return }
        guard delta.isFinite else { return }
        state.currentRotation += delta
        lassoTransformState = state
        applyLassoTransform(state)
    }

    func endLassoTransformIfNeeded() {
        lassoTransformState = nil
    }

    private func applyLassoTransform(_ state: LassoTransformState) {
        guard var selection = lassoSelection else { return }

        let cosR = cos(state.currentRotation)
        let sinR = sin(state.currentRotation)
        let scale = max(state.currentScale, 1e-6)

        func transformPoint(_ point: SIMD2<Double>) -> SIMD2<Double> {
            let dx = point.x - state.baseCenter.x
            let dy = point.y - state.baseCenter.y
            let sx = dx * scale
            let sy = dy * scale
            let rx = sx * cosR - sy * sinR
            let ry = sx * sinR + sy * cosR
            return SIMD2<Double>(state.baseCenter.x + rx, state.baseCenter.y + ry)
        }

        let transformedSelection = state.basePoints.map(transformPoint)
        selection.points = transformedSelection
        selection.bounds = polygonBounds(transformedSelection)
        selection.center = state.baseCenter
        lassoSelection = selection
        updateLassoPreview(for: selection)

        for snapshot in state.baseStrokes {
            guard snapshot.index < snapshot.frame.strokes.count else { continue }

            let transformedActivePoints = snapshot.activePoints.map(transformPoint)
            let transformedFramePoints = transformedActivePoints.map {
                SIMD2<Double>(
                    $0.x * snapshot.frameScale + snapshot.frameTranslation.x,
                    $0.y * snapshot.frameScale + snapshot.frameTranslation.y
                )
            }

            guard let first = transformedFramePoints.first else { continue }
            let origin = first
            let localPoints: [SIMD2<Float>] = transformedFramePoints.map {
                SIMD2<Float>(Float($0.x - origin.x), Float($0.y - origin.y))
            }

            let segments = Stroke.buildSegments(from: localPoints, color: snapshot.color)
            let bounds = Stroke.calculateBounds(for: localPoints, radius: Float(snapshot.worldWidth * scale) * 0.5)
            let newStroke = Stroke(
                id: snapshot.id,
                origin: origin,
                worldWidth: snapshot.worldWidth * scale,
                color: snapshot.color,
                zoomEffectiveAtCreation: snapshot.zoomEffectiveAtCreation,
                segments: segments,
                localBounds: bounds,
                segmentBounds: bounds,
                device: device,
                depthID: snapshot.depthID,
                depthWriteEnabled: snapshot.depthWriteEnabled
            )
            snapshot.frame.strokes[snapshot.index] = newStroke
        }

        for cardSnapshot in state.baseCards {
            if cardSnapshot.card.isLocked { continue }
            let transformedOriginActive = transformPoint(cardSnapshot.originActive)
            let originFrame = SIMD2<Double>(
                transformedOriginActive.x * cardSnapshot.frameScale + cardSnapshot.frameTranslation.x,
                transformedOriginActive.y * cardSnapshot.frameScale + cardSnapshot.frameTranslation.y
            )

            cardSnapshot.card.origin = originFrame
            cardSnapshot.card.rotation = cardSnapshot.rotation + Float(state.currentRotation)
            cardSnapshot.card.size = SIMD2<Double>(
                cardSnapshot.size.x * scale,
                cardSnapshot.size.y * scale
            )
            cardSnapshot.card.rebuildGeometry()
        }

        for snapshot in state.baseCardStrokes {
            if snapshot.card.isLocked { continue }
            guard snapshot.index < snapshot.card.strokes.count else { continue }

            let transformedActivePoints = snapshot.activePoints.map(transformPoint)
            let transformedFramePoints = transformedActivePoints.map {
                SIMD2<Double>(
                    $0.x * snapshot.frameScale + snapshot.frameTranslation.x,
                    $0.y * snapshot.frameScale + snapshot.frameTranslation.y
                )
            }

            let cInv = cos(-snapshot.cardRotation)
            let sInv = sin(-snapshot.cardRotation)
            let transformedCardLocal = transformedFramePoints.map {
                let dx = $0.x - snapshot.cardOrigin.x
                let dy = $0.y - snapshot.cardOrigin.y
                let localX = dx * cInv - dy * sInv
                let localY = dx * sInv + dy * cInv
                return SIMD2<Double>(localX, localY)
            }

            guard let first = transformedCardLocal.first else { continue }
            let origin = first
            let localPoints: [SIMD2<Float>] = transformedCardLocal.map {
                SIMD2<Float>(Float($0.x - origin.x), Float($0.y - origin.y))
            }

            let segments = Stroke.buildSegments(from: localPoints, color: snapshot.color)
            let bounds = Stroke.calculateBounds(for: localPoints, radius: Float(snapshot.worldWidth * scale) * 0.5)
            let newStroke = Stroke(
                id: snapshot.card.strokes[snapshot.index].id,
                origin: origin,
                worldWidth: snapshot.worldWidth * scale,
                color: snapshot.color,
                zoomEffectiveAtCreation: snapshot.zoomEffectiveAtCreation,
                segments: segments,
                localBounds: bounds,
                segmentBounds: bounds,
                device: device,
                depthID: snapshot.depthID,
                depthWriteEnabled: snapshot.depthWriteEnabled
            )
            snapshot.card.strokes[snapshot.index] = newStroke
        }
    }

    private func updateLassoPreview(for points: [SIMD2<Double>],
                                    close: Bool,
                                    card: Card? = nil,
                                    frame: Frame? = nil,
                                    zoom: Double? = nil) {
        let resolvedZoom = zoom ?? max(zoomScale, 1e-6)
        lassoPreviewStroke = buildLassoPreviewStroke(points: points, close: close, zoom: resolvedZoom)
        lassoPreviewFrame = activeFrame
        lassoPreviewCard = card
        lassoPreviewCardFrame = frame
    }

    private func updateLassoPreview(for selection: LassoSelection) {
        if let cardSelection = selection.cardStrokes.first {
            guard let transform = transformFromActive(to: cardSelection.frame) else {
                updateLassoPreview(for: selection.points, close: true)
                return
            }
            let cardPoints = selection.points.map { activePoint in
                let framePoint = SIMD2<Double>(
                    activePoint.x * transform.scale + transform.translation.x,
                    activePoint.y * transform.scale + transform.translation.y
                )
                return framePointToCardLocal(framePoint, card: cardSelection.card)
            }
            let zoomInFrame = zoomScale / max(transform.scale, 1e-6)
            updateLassoPreview(for: cardPoints,
                               close: true,
                               card: cardSelection.card,
                               frame: cardSelection.frame,
                               zoom: zoomInFrame)
        } else {
            updateLassoPreview(for: selection.points, close: true)
        }
    }

    private func buildLassoPreviewStroke(points: [SIMD2<Double>], close: Bool, zoom: Double) -> Stroke? {
        guard points.count >= 2 else { return nil }

        var path = points
        if close, let first = points.first, let last = points.last {
            let dx = last.x - first.x
            let dy = last.y - first.y
            if (dx * dx + dy * dy) > 1e-12 {
                path.append(first)
            }
        }

        let safeZoom = max(zoom, 1e-6)
        let dashWorld = lassoDashLengthPx / safeZoom
        let gapWorld = lassoGapLengthPx / safeZoom
        let widthWorld = lassoLineWidthPx / safeZoom

        let origin = path[0]
        var segments: [StrokeSegmentInstance] = []
        var boundPoints: [SIMD2<Float>] = []

        for i in 0..<(path.count - 1) {
            let a = path[i]
            let b = path[i + 1]
            let dx = b.x - a.x
            let dy = b.y - a.y
            let len = sqrt(dx * dx + dy * dy)
            if len <= 0 { continue }

            let ux = dx / len
            let uy = dy / len
            var t = 0.0
            while t < len {
                let segLen = min(dashWorld, len - t)
                let start = t
                let end = t + segLen

                let p0World = SIMD2<Double>(a.x + ux * start, a.y + uy * start)
                let p1World = SIMD2<Double>(a.x + ux * end, a.y + uy * end)

                let p0Local = SIMD2<Float>(Float(p0World.x - origin.x), Float(p0World.y - origin.y))
                let p1Local = SIMD2<Float>(Float(p1World.x - origin.x), Float(p1World.y - origin.y))

                segments.append(StrokeSegmentInstance(p0: p0Local, p1: p1Local, color: lassoColor))
                boundPoints.append(p0Local)
                boundPoints.append(p1Local)

                t += dashWorld + gapWorld
            }
        }

        guard !segments.isEmpty else { return nil }
        let bounds = Stroke.calculateBounds(for: boundPoints, radius: Float(widthWorld) * 0.5)
        return Stroke(
            id: UUID(),
            origin: origin,
            worldWidth: widthWorld,
            color: lassoColor,
            zoomEffectiveAtCreation: Float(max(zoomScale, 1e-6)),
            segments: segments,
            localBounds: bounds,
            segmentBounds: bounds,
            device: device,
            depthID: Self.strokeDepthSlotCount - 1,
            depthWriteEnabled: false
        )
    }

    private func polygonBounds(_ points: [SIMD2<Double>]) -> CGRect {
        guard let first = points.first else { return .null }
        var minX = first.x
        var maxX = first.x
        var minY = first.y
        var maxY = first.y

        for p in points.dropFirst() {
            minX = min(minX, p.x)
            maxX = max(maxX, p.x)
            minY = min(minY, p.y)
            maxY = max(maxY, p.y)
        }

        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }

    private func framePointToCardLocal(_ point: SIMD2<Double>, card: Card) -> SIMD2<Double> {
        let dx = point.x - card.origin.x
        let dy = point.y - card.origin.y
        let c = cos(Double(-card.rotation))
        let s = sin(Double(-card.rotation))
        let localX = dx * c - dy * s
        let localY = dx * s + dy * c
        return SIMD2<Double>(localX, localY)
    }

    /// Check if a point in frame coordinates is on the card's resize handle (bottom-right corner)
    /// Returns true if the point is within the handle hit area
    func isPointOnCardHandle(_ pointInFrame: SIMD2<Double>, card: Card, zoom: Double) -> Bool {
        // Convert to card-local coordinates
        let localPoint = framePointToCardLocal(pointInFrame, card: card)

        // Handle is in bottom-right corner
        let halfW = card.size.x / 2.0
        let halfH = card.size.y / 2.0

        // Define handle hit area size (in world units)
        // Make it screen-size independent by dividing by zoom
        let handleSizePx: Double = 40.0 // 40px hit area
        let handleSize = handleSizePx / max(zoom, 1e-6)

        // Check if point is in bottom-right corner region
        let isInRightEdge = localPoint.x >= (halfW - handleSize) && localPoint.x <= halfW
        let isInBottomEdge = localPoint.y >= (halfH - handleSize) && localPoint.y <= halfH

        return isInRightEdge && isInBottomEdge
    }

    private func cardOffsetFromCameraInLocalSpace(card: Card, cameraCenter: SIMD2<Double>) -> SIMD2<Double> {
        let cameraLocal = framePointToCardLocal(cameraCenter, card: card)
        return SIMD2<Double>(-cameraLocal.x, -cameraLocal.y)
    }

    private func shadowOffsetInCardLocalSpace(offsetPx: SIMD2<Float>,
                                              rotation: Float,
                                              zoom: Double) -> SIMD2<Double> {
        let dx = Double(offsetPx.x) / max(zoom, 1e-6)
        let dy = Double(offsetPx.y) / max(zoom, 1e-6)
        let angle = Double(rotation)
        let c = cos(angle)
        let s = sin(angle)
        let localX = dx * c + dy * s
        let localY = -dx * s + dy * c
        return SIMD2<Double>(localX, localY)
    }

    private func cardLocalToFramePoint(_ point: SIMD2<Double>, card: Card) -> SIMD2<Double> {
        let c = cos(Double(card.rotation))
        let s = sin(Double(card.rotation))
        let rotX = point.x * c - point.y * s
        let rotY = point.x * s + point.y * c
        return SIMD2<Double>(card.origin.x + rotX, card.origin.y + rotY)
    }

    private func framesInActiveChain() -> [Frame] {
        var frames: [Frame] = []

        var current: Frame? = activeFrame
        while let frame = current {
            frames.append(frame)
            current = frame.parent
        }

        current = childFrame(of: activeFrame)
        while let frame = current {
            frames.append(frame)
            current = childFrame(of: frame)
        }

        return frames
    }

    private func pointInPolygon(_ point: SIMD2<Double>, polygon: [SIMD2<Double>]) -> Bool {
        guard polygon.count >= 3 else { return false }
        var inside = false
        var j = polygon.count - 1
        for i in 0..<polygon.count {
            let pi = polygon[i]
            let pj = polygon[j]
            let denom = pj.y - pi.y
            if abs(denom) > 1e-12 {
                let intersect = ((pi.y > point.y) != (pj.y > point.y)) &&
                    (point.x < (pj.x - pi.x) * (point.y - pi.y) / denom + pi.x)
                if intersect {
                    inside.toggle()
                }
            }
            j = i
        }
        return inside
    }

    private func polygonEdges(_ points: [SIMD2<Double>]) -> [(SIMD2<Double>, SIMD2<Double>)] {
        guard points.count >= 2 else { return [] }
        var edges: [(SIMD2<Double>, SIMD2<Double>)] = []
        edges.reserveCapacity(points.count)
        for i in 0..<(points.count - 1) {
            edges.append((points[i], points[i + 1]))
        }
        if let first = points.first, let last = points.last, first != last {
            edges.append((last, first))
        }
        return edges
    }

    private func segmentsIntersect(_ p1: SIMD2<Double>, _ q1: SIMD2<Double>, _ p2: SIMD2<Double>, _ q2: SIMD2<Double>) -> Bool {
        let eps = 1e-12

        func cross(_ a: SIMD2<Double>, _ b: SIMD2<Double>, _ c: SIMD2<Double>) -> Double {
            (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
        }

        func onSegment(_ a: SIMD2<Double>, _ b: SIMD2<Double>, _ c: SIMD2<Double>) -> Bool {
            min(a.x, c.x) - eps <= b.x && b.x <= max(a.x, c.x) + eps &&
            min(a.y, c.y) - eps <= b.y && b.y <= max(a.y, c.y) + eps
        }

        let o1 = cross(p1, q1, p2)
        let o2 = cross(p1, q1, q2)
        let o3 = cross(p2, q2, p1)
        let o4 = cross(p2, q2, q1)

        if (o1 * o2 < 0) && (o3 * o4 < 0) {
            return true
        }

        if abs(o1) <= eps && onSegment(p1, p2, q1) { return true }
        if abs(o2) <= eps && onSegment(p1, q2, q1) { return true }
        if abs(o3) <= eps && onSegment(p2, p1, q2) { return true }
        if abs(o4) <= eps && onSegment(p2, q1, q2) { return true }

        return false
    }

    private func strokeIntersectsPolygon(_ stroke: Stroke,
                                         polygon: [SIMD2<Double>],
                                         polygonBounds: CGRect,
                                         edges: [(SIMD2<Double>, SIMD2<Double>)]) -> Bool {
        let sBounds = stroke.localBounds
        if sBounds == .null { return false }

        let strokeBoundsWorld = CGRect(
            x: stroke.origin.x + Double(sBounds.minX),
            y: stroke.origin.y + Double(sBounds.minY),
            width: Double(sBounds.width),
            height: Double(sBounds.height)
        )

        if !strokeBoundsWorld.intersects(polygonBounds) {
            return false
        }

        for seg in stroke.segments {
            let p0 = SIMD2<Double>(stroke.origin.x + Double(seg.p0.x),
                                   stroke.origin.y + Double(seg.p0.y))
            let p1 = SIMD2<Double>(stroke.origin.x + Double(seg.p1.x),
                                   stroke.origin.y + Double(seg.p1.y))

            if pointInPolygon(p0, polygon: polygon) || pointInPolygon(p1, polygon: polygon) {
                return true
            }

            for edge in edges {
                if segmentsIntersect(p0, p1, edge.0, edge.1) {
                    return true
                }
            }
        }

        return false
    }

    private func selectStrokes(from strokes: [Stroke], polygon: [SIMD2<Double>]) -> Set<UUID> {
        guard polygon.count >= 3 else { return [] }
        let bounds = polygonBounds(polygon)
        let edges = polygonEdges(polygon)

        var ids = Set<UUID>()
        for stroke in strokes {
            if strokeIntersectsPolygon(stroke, polygon: polygon, polygonBounds: bounds, edges: edges) {
                ids.insert(stroke.id)
            }
        }
        return ids
    }

    private func selectStrokes(in frame: Frame, polygon: [SIMD2<Double>]) -> Set<UUID> {
        selectStrokes(from: frame.strokes, polygon: polygon)
    }

    private func polygonIntersectsCard(_ polygonActive: [SIMD2<Double>],
                                       card: Card,
                                       transform: (scale: Double, translation: SIMD2<Double>)) -> Bool {
        guard polygonActive.count >= 3 else { return false }
        let polygonCard = polygonActive.map { activePoint in
            let framePoint = SIMD2<Double>(
                activePoint.x * transform.scale + transform.translation.x,
                activePoint.y * transform.scale + transform.translation.y
            )
            return framePointToCardLocal(framePoint, card: card)
        }

        let bounds = polygonBounds(polygonCard)
        let halfW = card.size.x * 0.5
        let halfH = card.size.y * 0.5
        let cardRect = CGRect(x: -halfW, y: -halfH, width: card.size.x, height: card.size.y)
        if !bounds.intersects(cardRect) {
            return false
        }

        // Any polygon point inside card rect?
        if polygonCard.contains(where: { cardRect.contains(CGPoint(x: $0.x, y: $0.y)) }) {
            return true
        }

        // Any card corner inside polygon?
        let corners = [
            SIMD2<Double>(-halfW, -halfH),
            SIMD2<Double>(halfW, -halfH),
            SIMD2<Double>(halfW, halfH),
            SIMD2<Double>(-halfW, halfH)
        ]
        if corners.contains(where: { pointInPolygon($0, polygon: polygonCard) }) {
            return true
        }

        // Edge intersection test.
        let polygonEdges = polygonEdges(polygonCard)
        let cardEdges: [(SIMD2<Double>, SIMD2<Double>)] = [
            (corners[0], corners[1]),
            (corners[1], corners[2]),
            (corners[2], corners[3]),
            (corners[3], corners[0])
        ]

        for edge in polygonEdges {
            for cardEdge in cardEdges {
                if segmentsIntersect(edge.0, edge.1, cardEdge.0, cardEdge.1) {
                    return true
                }
            }
        }

        return false
    }

	    // MARK: - Touch Handling

	    private var isRunningOnMac: Bool {
	#if targetEnvironment(macCatalyst)
	        return true
	#else
	        if #available(iOS 14.0, *) {
	            return ProcessInfo.processInfo.isiOSAppOnMac
	        }
	        return false
	#endif
	    }

	    private func isDrawingTouchType(_ touchType: UITouch.TouchType) -> Bool {
	        if touchType == .pencil { return true }
	        guard isRunningOnMac else { return false }

	        // iOS app on Mac / Mac Catalyst: allow mouse/trackpad drawing.
	        if #available(iOS 13.4, macCatalyst 13.4, *) {
	            return touchType == .indirectPointer || touchType == .direct
	        }
	        return touchType == .direct
	    }
	
    func handleTouchBegan(at point: CGPoint, touchType: UITouch.TouchType) {
        // MODAL INPUT: Pencil on iPad; mouse/trackpad on Mac Catalyst.
        guard isDrawingTouchType(touchType) else { return }

        guard let view = metalView else { return }

        if brushSettings.isLasso {
            clearLassoSelection()
            if let (card, frame, _, _) = hitTestHierarchy(screenPoint: point, viewSize: view.bounds.size, ignoringLocked: true) {
                lassoTarget = .card(card, frame)
            } else {
                lassoTarget = .canvas
            }
            lassoDrawingPoints = [point]
            lassoPredictedPoints = []
            if let screenPoints = buildLassoScreenPoints() {
                updateLassoPreviewFromScreenPoints(screenPoints, close: false, viewSize: view.bounds.size)
            }
            currentTouchPoints = []
            predictedTouchPoints = []
            liveStrokeOrigin = nil
            currentDrawingTarget = nil
            lastSavedPoint = nil
            return
        }

        if brushSettings.isStrokeEraser {
            eraseStrokeAtPoint(screenPoint: point, viewSize: view.bounds.size)
            currentTouchPoints = []
            predictedTouchPoints = []
            liveStrokeOrigin = nil
            currentDrawingTarget = nil
            lastSavedPoint = nil
            return
        }

        if brushSettings.isMaskEraser {
            if let (card, frame, _, _) = hitTestHierarchy(screenPoint: point, viewSize: view.bounds.size, ignoringLocked: true) {
                // If we're over a card, always target that card's strokes.
                currentDrawingTarget = .card(card, frame)
                liveStrokeOrigin = screenToWorldPixels_PureDouble(
                    point,
                    viewSize: view.bounds.size,
                    panOffset: panOffset,
                    zoomScale: zoomScale,
                    rotationAngle: rotationAngle
                )
            } else if let strokeHit = hitTestStrokeHierarchy(screenPoint: point, viewSize: view.bounds.size) {
                switch strokeHit.target {
                case .canvas(let frame, let pointInFrame):
                    currentDrawingTarget = .canvas(frame)
                    liveStrokeOrigin = pointInFrame
                case .card(let card, let frame):
                    currentDrawingTarget = .card(card, frame)
                    liveStrokeOrigin = screenToWorldPixels_PureDouble(
                        point,
                        viewSize: view.bounds.size,
                        panOffset: panOffset,
                        zoomScale: zoomScale,
                        rotationAngle: rotationAngle
                    )
                }
            } else {
                currentDrawingTarget = .canvas(activeFrame)
                liveStrokeOrigin = screenToWorldPixels_PureDouble(
                    point,
                    viewSize: view.bounds.size,
                    panOffset: panOffset,
                    zoomScale: zoomScale,
                    rotationAngle: rotationAngle
                )
            }
        } else {
            // USE HIERARCHICAL CARD HIT TEST
            if let (card, frame, _, _) = hitTestHierarchy(screenPoint: point, viewSize: view.bounds.size, ignoringLocked: true) {
                // Found a card (in active, parent, or child frame)
                // Store BOTH the card AND the frame it belongs to for correct coordinate transforms
                currentDrawingTarget = .card(card, frame)

                // For live rendering, we need the origin in ACTIVE World Space
                liveStrokeOrigin = screenToWorldPixels_PureDouble(
                    point,
                    viewSize: view.bounds.size,
                    panOffset: panOffset,
                    zoomScale: zoomScale,
                    rotationAngle: rotationAngle
                )

            } else {
                // Draw on Canvas (Active Frame)
                currentDrawingTarget = .canvas(activeFrame)

                liveStrokeOrigin = screenToWorldPixels_PureDouble(
                    point,
                    viewSize: view.bounds.size,
                    panOffset: panOffset,
                    zoomScale: zoomScale,
                    rotationAngle: rotationAngle
                )
            }
        }

        // Keep points in SCREEN space during drawing
        currentTouchPoints = [point]
    }

    func handleTouchMoved(at point: CGPoint, predicted: [CGPoint], touchType: UITouch.TouchType) {
        // MODAL INPUT: Pencil on iPad; mouse/trackpad on Mac Catalyst.
        guard isDrawingTouchType(touchType) else { return }

        if brushSettings.isLasso {
            let minimumDistance: CGFloat = 2.0
            var shouldAdd = false
            if let last = lassoDrawingPoints.last {
                let dist = hypot(point.x - last.x, point.y - last.y)
                if dist > minimumDistance {
                    shouldAdd = true
                }
            } else {
                shouldAdd = true
            }

            if shouldAdd {
                lassoDrawingPoints.append(point)
            }

            lassoPredictedPoints = predicted
            if let view = metalView, let screenPoints = buildLassoScreenPoints() {
                updateLassoPreviewFromScreenPoints(screenPoints, close: false, viewSize: view.bounds.size)
            }
            return
        }

        if brushSettings.isStrokeEraser {
            guard let view = metalView else { return }
            eraseStrokeAtPoint(screenPoint: point, viewSize: view.bounds.size)
            return
        }

        //  OPTIMIZATION: Distance Filter
        // Only add the point if it's far enough from the last one (e.g., 2.0 pixels)
        // This prevents "vertex explosion" when drawing slowly at 120Hz/240Hz.
        let minimumDistance: CGFloat = 2.0

        var shouldAdd = false
        if let last = currentTouchPoints.last {
            let dist = hypot(point.x - last.x, point.y - last.y)
            if dist > minimumDistance {
                shouldAdd = true
            }
        } else {
            shouldAdd = true // Always add the first point
        }

        if shouldAdd {
            currentTouchPoints.append(point)
            lastSavedPoint = point
        }

        // Update prediction (always do this for responsiveness, even if we filter the real point)
        predictedTouchPoints = predicted
    }

    func handleTouchEnded(at point: CGPoint, touchType: UITouch.TouchType) {
        // MODAL INPUT: Pencil on iPad; mouse/trackpad on Mac Catalyst.
        guard isDrawingTouchType(touchType) else { return }
        guard let view = metalView else { return }

        if brushSettings.isLasso {
            lassoPredictedPoints = []
            lassoDrawingPoints.append(point)
            let rawScreenPoints = lassoDrawingPoints
            lassoDrawingPoints = []

            let simplified = simplifyStroke(rawScreenPoints, minScreenDist: 2.0, minAngleDeg: 5.0)
            guard simplified.count >= 3 else {
                clearLassoSelection()
                return
            }

            let worldPoints = simplified.map {
                screenToWorldPixels_PureDouble(
                    $0,
                    viewSize: view.bounds.size,
                    panOffset: panOffset,
                    zoomScale: zoomScale,
                    rotationAngle: rotationAngle
                )
            }

            var closedPoints = worldPoints
            if let first = worldPoints.first, let last = worldPoints.last {
                let dx = last.x - first.x
                let dy = last.y - first.y
                if (dx * dx + dy * dy) > 1e-12 {
                    closedPoints.append(first)
                }
            }

            let bounds = polygonBounds(closedPoints)
            let center = SIMD2<Double>(Double(bounds.midX), Double(bounds.midY))

        switch lassoTarget {
        case .card(let card, let frame):
            guard !card.isLocked else {
                clearLassoSelection()
                lassoTarget = nil
                return
            }
            guard let transform = transformFromActive(to: frame) else {
                clearLassoSelection()
                return
            }

                let polygonInCard = closedPoints.map { activePoint in
                    let framePoint = SIMD2<Double>(
                        activePoint.x * transform.scale + transform.translation.x,
                        activePoint.y * transform.scale + transform.translation.y
                    )
                    return framePointToCardLocal(framePoint, card: card)
                }

                let selectedIDs = selectStrokes(from: card.strokes, polygon: polygonInCard)

                lassoSelection = LassoSelection(
                    points: closedPoints,
                    bounds: bounds,
                    center: center,
                    frames: [],
                    cards: [],
                    cardStrokes: [LassoCardStrokeSelection(card: card, frame: frame, strokeIDs: selectedIDs)]
                )

            default:
                var selections: [LassoFrameSelection] = []
                var cardSelections: [LassoCardSelection] = []

                for frame in framesInActiveChain() {
                    guard let transform = transformFromActive(to: frame) else { continue }
                    let polygonInFrame = closedPoints.map {
                        SIMD2<Double>(
                            $0.x * transform.scale + transform.translation.x,
                            $0.y * transform.scale + transform.translation.y
                        )
                    }
                    let selectedIDs = selectStrokes(in: frame, polygon: polygonInFrame)
                    if !selectedIDs.isEmpty {
                        selections.append(LassoFrameSelection(frame: frame, strokeIDs: selectedIDs))
                    }

                    for card in frame.cards where !card.isLocked {
                        if polygonIntersectsCard(closedPoints, card: card, transform: transform) {
                            cardSelections.append(LassoCardSelection(card: card, frame: frame))
                        }
                    }
                }

                lassoSelection = LassoSelection(
                    points: closedPoints,
                    bounds: bounds,
                    center: center,
                    frames: selections,
                    cards: cardSelections,
                    cardStrokes: []
                )
            }

            if let selection = lassoSelection {
                updateLassoPreview(for: selection)
            }
            lassoTarget = nil
            return
        }

        if brushSettings.isStrokeEraser {
            eraseStrokeAtPoint(screenPoint: point, viewSize: view.bounds.size)
            predictedTouchPoints = []
            currentTouchPoints = []
            liveStrokeOrigin = nil
            currentDrawingTarget = nil
            lastSavedPoint = nil
            return
        }

        guard let target = currentDrawingTarget else { return }

        // Clear predictions (no longer needed)
        predictedTouchPoints = []

        // Keep final point in SCREEN space
        currentTouchPoints.append(point)

        //  FIX 1: Allow dots (Don't return if count < 4)
        guard !currentTouchPoints.isEmpty else {
            currentTouchPoints = []
            liveStrokeOrigin = nil
            currentDrawingTarget = nil
            return
        }

        //  FIX 2: Phantom Points for Catmull-Rom
        // Extrapolate phantom points to ensure the spline reaches the start and end
        var smoothScreenPoints: [CGPoint]

        if currentTouchPoints.count < 3 {
            // Too few points for a spline, just use lines/dots
            smoothScreenPoints = currentTouchPoints
        } else {
            // Extrapolate phantom points instead of duplicating
            // First phantom: A - (B - A) = 2A - B (extends backward from A)
            // Last phantom: D + (D - C) = 2D - C (extends forward from D)
            var paddedPoints = currentTouchPoints

            // Add phantom point at the start
            if paddedPoints.count >= 2 {
                let first = paddedPoints[0]
                let second = paddedPoints[1]
                let phantomStart = CGPoint(
                    x: 2 * first.x - second.x,
                    y: 2 * first.y - second.y
                )
                paddedPoints.insert(phantomStart, at: 0)
            }

            // Add phantom point at the end
            if paddedPoints.count >= 3 {  // Now at least 3 because we added one
                let last = paddedPoints[paddedPoints.count - 1]
                let secondLast = paddedPoints[paddedPoints.count - 2]
                let phantomEnd = CGPoint(
                    x: 2 * last.x - secondLast.x,
                    y: 2 * last.y - secondLast.y
                )
                paddedPoints.append(phantomEnd)
            }

            smoothScreenPoints = catmullRomPoints(points: paddedPoints,
                                                  closed: false,
                                                  alpha: 0.5,
                                                  segmentsPerCurve: 20)

            // Apply simplification to reduce vertex count
            smoothScreenPoints = simplifyStroke(
                smoothScreenPoints,
                minScreenDist: 1.5,
                minAngleDeg: 5.0
            )
        }

        //  MODAL INPUT: Route stroke to correct target
        let isMaskEraser = brushSettings.isMaskEraser
        let strokeColor = isMaskEraser ? SIMD4<Float>(0, 0, 0, 0) : brushSettings.color
        let strokeDepthWriteEnabled = isMaskEraser ? true : brushSettings.depthWriteEnabled
        switch target {
        case .canvas(let frame):
            if frame === activeFrame {
                // DRAW ON CANVAS (Active Frame)
	                let stroke = Stroke(screenPoints: smoothScreenPoints,
	                                    zoomAtCreation: zoomScale,
	                                    panAtCreation: panOffset,
	                                    viewSize: view.bounds.size,
	                                    rotationAngle: rotationAngle,
	                                    color: strokeColor,
	                                    baseWidth: brushSettings.size,
	                                    zoomEffectiveAtCreation: Float(max(zoomScale, 1e-6)),
	                                    device: device,
	                                    depthID: allocateStrokeDepthID(),
	                                    depthWriteEnabled: strokeDepthWriteEnabled,
	                                    constantScreenSize: brushSettings.constantScreenSize)
	                frame.strokes.append(stroke)
	                pushUndo(.drawStroke(stroke: stroke, target: target))
            } else {
                // DRAW ON CANVAS (Other Frame in Telescope Chain)
                let stroke = createStrokeForFrame(
                    screenPoints: smoothScreenPoints,
                    frame: frame,
                    viewSize: view.bounds.size,
                    depthID: allocateStrokeDepthID(),
                    color: strokeColor,
                    depthWriteEnabled: strokeDepthWriteEnabled
                )
                frame.strokes.append(stroke)
                pushUndo(.drawStroke(stroke: stroke, target: target))
            }

        case .card(let card, let frame):
            // DRAW ON CARD (Cross-Depth Compatible)
            // Transform points into card-local space accounting for which frame the card is in
	            let cardStroke = createStrokeForCard(
	                screenPoints: smoothScreenPoints,
	                card: card,
	                frame: frame,
	                viewSize: view.bounds.size,
	                depthID: allocateStrokeDepthID(),
	                color: strokeColor,
	                depthWriteEnabled: strokeDepthWriteEnabled
	            )
	            card.strokes.append(cardStroke)
	            pushUndo(.drawStroke(stroke: cardStroke, target: target))
        }

        currentTouchPoints = []
        liveStrokeOrigin = nil
        currentDrawingTarget = nil
        lastSavedPoint = nil  // Clear for next stroke
    }

    func handleTouchCancelled(touchType: UITouch.TouchType) {
        // MODAL INPUT: Pencil on iPad; mouse/trackpad on Mac Catalyst.
        guard isDrawingTouchType(touchType) else { return }
        if brushSettings.isLasso {
            lassoDrawingPoints = []
            lassoPredictedPoints = []
            if let selection = lassoSelection {
                updateLassoPreview(for: selection)
            } else {
                lassoPreviewStroke = nil
                lassoPreviewFrame = nil
            }
            return
        }
        predictedTouchPoints = []  // Clear predictions
        currentTouchPoints = []
        liveStrokeOrigin = nil  // Clear temporary origin
        lastSavedPoint = nil  // Clear for next stroke
    }

    // MARK: - Cross-Frame Hit Testing (Linked List Frames)

    private struct FrameTransform {
        let frame: Frame
        let point: SIMD2<Double>
        let scale: Double  // Active -> Frame scale factor
    }

    private struct StrokeHit {
        enum Target {
            case canvas(frame: Frame, pointInFrame: SIMD2<Double>)
            case card(card: Card, frame: Frame)
        }

        let target: Target
        let stroke: Stroke
        let depthID: UInt32
    }

    private func childFrame(of frame: Frame) -> Frame? {
        // Frames are a linked list: at most one child.
        frame.children.first
    }

    private func collectFrameTransforms(pointActive: SIMD2<Double>) -> (ancestors: [FrameTransform], descendants: [FrameTransform]) {
        var ancestors: [FrameTransform] = []
        var descendants: [FrameTransform] = []

        // Walk up to parents.
        var current: Frame? = activeFrame
        var point = pointActive
        var scale = 1.0
        while let frame = current, let parent = frame.parent {
            point = frame.originInParent + (point / frame.scaleRelativeToParent)
            scale /= frame.scaleRelativeToParent
            ancestors.append(FrameTransform(frame: parent, point: point, scale: scale))
            current = parent
        }

        // Walk down to child chain.
        current = activeFrame
        point = pointActive
        scale = 1.0
        while let frame = current, let child = childFrame(of: frame) {
            point = (point - child.originInParent) * child.scaleRelativeToParent
            scale *= child.scaleRelativeToParent
            descendants.append(FrameTransform(frame: child, point: point, scale: scale))
            current = child
        }

        return (ancestors, descendants)
    }

    private func transformFromActive(to target: Frame) -> (scale: Double, translation: SIMD2<Double>)? {
        if target === activeFrame {
            return (scale: 1.0, translation: .zero)
        }

        // Walk up to find ancestor.
        var scale = 1.0
        var translation = SIMD2<Double>(repeating: 0.0)
        var current: Frame? = activeFrame
        while let frame = current, let parent = frame.parent {
            scale /= frame.scaleRelativeToParent
            translation = translation / frame.scaleRelativeToParent + frame.originInParent
            if parent === target {
                return (scale, translation)
            }
            current = parent
        }

        // Walk down to find descendant.
        scale = 1.0
        translation = SIMD2<Double>(repeating: 0.0)
        current = activeFrame
        while let frame = current, let child = childFrame(of: frame) {
            scale *= child.scaleRelativeToParent
            translation = (translation - child.originInParent) * child.scaleRelativeToParent
            if child === target {
                return (scale, translation)
            }
            current = child
        }

        return nil
    }

    private func pointInFrame(screenPoint: CGPoint, viewSize: CGSize, frame: Frame) -> SIMD2<Double>? {
        let pointActive = screenToWorldPixels_PureDouble(
            screenPoint,
            viewSize: viewSize,
            panOffset: panOffset,
            zoomScale: zoomScale,
            rotationAngle: rotationAngle
        )

        if frame === activeFrame {
            return pointActive
        }

        guard let transform = transformFromActive(to: frame) else { return nil }
        return pointActive * transform.scale + transform.translation
    }

    private func eraserRadiusWorld(forScale scale: Double) -> Double {
        let safeZoom = max(zoomScale, 1e-6)
        return (brushSettings.size * 0.5) * scale / safeZoom
    }

    private func hitTestStroke(_ stroke: Stroke,
                               pointInFrame: SIMD2<Double>,
                               eraserRadius: Double) -> Bool {
        let localX = pointInFrame.x - stroke.origin.x
        let localY = pointInFrame.y - stroke.origin.y

        if stroke.localBounds != .null {
            // Broad-phase: skip per-segment math if eraser bounds don't overlap stroke bounds.
            let sBounds = stroke.localBounds
            let eMinX = localX - eraserRadius
            let eMaxX = localX + eraserRadius
            let eMinY = localY - eraserRadius
            let eMaxY = localY + eraserRadius

            if eMaxX < Double(sBounds.minX) ||
                eMinX > Double(sBounds.maxX) ||
                eMaxY < Double(sBounds.minY) ||
                eMinY > Double(sBounds.maxY) {
                return false
            }
        }

        let radius = Float(stroke.worldWidth * 0.5 + eraserRadius)
        let radiusSq = radius * radius
        let p = SIMD2<Float>(Float(localX), Float(localY))

        for seg in stroke.segments {
            let a = seg.p0
            let b = seg.p1
            let ab = b - a
            let ap = p - a
            let denom = simd_dot(ab, ab)
            let t = denom > 0 ? max(0.0, min(1.0, simd_dot(ap, ab) / denom)) : 0.0
            let closest = a + ab * t
            let d = p - closest
            let distSq = simd_dot(d, d)
            if distSq <= radiusSq {
                return true
            }
        }

        return false
    }

    private func hitTestCardStroke(card: Card,
                                   pointInFrame: SIMD2<Double>,
                                   eraserRadius: Double,
                                   minimumDepthID: UInt32? = nil) -> Stroke? {
        guard !card.isLocked else { return nil }
        guard !card.strokes.isEmpty else { return nil }
        guard card.hitTest(pointInFrame: pointInFrame) else { return nil }

        let dx = pointInFrame.x - card.origin.x
        let dy = pointInFrame.y - card.origin.y
        let c = cos(-card.rotation)
        let s = sin(-card.rotation)
        let localX = dx * Double(c) - dy * Double(s)
        let localY = dx * Double(s) + dy * Double(c)
        let pointInCard = SIMD2<Double>(localX, localY)

        for stroke in card.strokes.reversed() {
            if let minimumDepthID, stroke.depthID <= minimumDepthID {
                break
            }
            if hitTestStroke(stroke,
                             pointInFrame: pointInCard,
                             eraserRadius: eraserRadius) {
                return stroke
            }
        }

        return nil
    }

    private func hitTestStrokeHierarchy(screenPoint: CGPoint, viewSize: CGSize) -> StrokeHit? {
        let pointActive = screenToWorldPixels_PureDouble(
            screenPoint,
            viewSize: viewSize,
            panOffset: panOffset,
            zoomScale: zoomScale,
            rotationAngle: rotationAngle
        )

        let transforms = collectFrameTransforms(pointActive: pointActive)
        var bestDepthID: UInt32?
        var bestTarget: StrokeHit.Target?
        var bestStroke: Stroke?

        func considerCanvasStrokes(in frame: Frame,
                                   pointInFrame: SIMD2<Double>,
                                   eraserRadius: Double) {
            guard !frame.strokes.isEmpty else { return }
            for stroke in frame.strokes.reversed() {
                if let best = bestDepthID, stroke.depthID <= best {
                    break
                }
                if hitTestStroke(stroke,
                                 pointInFrame: pointInFrame,
                                 eraserRadius: eraserRadius) {
                    bestDepthID = stroke.depthID
                    bestTarget = .canvas(frame: frame, pointInFrame: pointInFrame)
                    bestStroke = stroke
                    break
                }
            }
        }

        func considerCardStrokes(in frame: Frame,
                                 pointInFrame: SIMD2<Double>,
                                 eraserRadius: Double) {
            guard !frame.cards.isEmpty else { return }

            for card in frame.cards.reversed() {
                if card.isLocked { continue }
                guard let newest = card.strokes.last?.depthID else { continue }
                if let best = bestDepthID, newest <= best {
                    continue
                }
                if let stroke = hitTestCardStroke(card: card,
                                                  pointInFrame: pointInFrame,
                                                  eraserRadius: eraserRadius,
                                                  minimumDepthID: bestDepthID) {
                    bestDepthID = stroke.depthID
                    bestTarget = .card(card: card, frame: frame)
                    bestStroke = stroke
                    break
                }
            }
        }

        // Active frame first, then ancestors/descendants.
        let activeEraserRadius = eraserRadiusWorld(forScale: 1.0)
        considerCanvasStrokes(in: activeFrame,
                              pointInFrame: pointActive,
                              eraserRadius: activeEraserRadius)
        considerCardStrokes(in: activeFrame,
                            pointInFrame: pointActive,
                            eraserRadius: activeEraserRadius)

        for ancestor in transforms.ancestors {
            let eraserRadius = eraserRadiusWorld(forScale: ancestor.scale)
            considerCanvasStrokes(in: ancestor.frame,
                                  pointInFrame: ancestor.point,
                                  eraserRadius: eraserRadius)
            considerCardStrokes(in: ancestor.frame,
                                pointInFrame: ancestor.point,
                                eraserRadius: eraserRadius)
        }

        for descendant in transforms.descendants {
            let eraserRadius = eraserRadiusWorld(forScale: descendant.scale)
            considerCanvasStrokes(in: descendant.frame,
                                  pointInFrame: descendant.point,
                                  eraserRadius: eraserRadius)
            considerCardStrokes(in: descendant.frame,
                                pointInFrame: descendant.point,
                                eraserRadius: eraserRadius)
        }

        guard let depthID = bestDepthID, let target = bestTarget, let stroke = bestStroke else { return nil }
        return StrokeHit(target: target, stroke: stroke, depthID: depthID)
    }

    private func eraseStrokeAtPoint(screenPoint: CGPoint, viewSize: CGSize) {
        // Cards sit on top of the canvas; when covered, only card strokes are eligible.
        if let (card, frame, conversionScale, _) = hitTestHierarchy(screenPoint: screenPoint, viewSize: viewSize, ignoringLocked: true) {
            guard let pointInTargetFrame = pointInFrame(screenPoint: screenPoint,
                                                        viewSize: viewSize,
                                                        frame: frame) else { return }
            let eraserRadius = eraserRadiusWorld(forScale: conversionScale)
            if let stroke = hitTestCardStroke(card: card,
                                              pointInFrame: pointInTargetFrame,
                                              eraserRadius: eraserRadius),
               let index = card.strokes.firstIndex(where: { $0 === stroke }) {
                let strokeCopy = stroke // Keep reference before removing
                card.strokes.remove(at: index)
                pushUndo(.eraseStroke(stroke: strokeCopy, strokeIndex: index, target: .card(card, frame)))
            }
            return
        }

        guard let hit = hitTestStrokeHierarchy(screenPoint: screenPoint, viewSize: viewSize) else { return }
        switch hit.target {
        case .canvas(let frame, _):
            if let index = frame.strokes.firstIndex(where: { $0 === hit.stroke }) {
                let strokeCopy = hit.stroke // Keep reference before removing
                frame.strokes.remove(at: index)
                pushUndo(.eraseStroke(stroke: strokeCopy, strokeIndex: index, target: .canvas(frame)))
            }
        case .card(let card, let frame):
            if let index = card.strokes.firstIndex(where: { $0 === hit.stroke }) {
                let strokeCopy = hit.stroke // Keep reference before removing
                card.strokes.remove(at: index)
                pushUndo(.eraseStroke(stroke: strokeCopy, strokeIndex: index, target: .card(card, frame)))
            }
        }
    }

    // MARK: - Card Management

    /// Hit test across the entire visible chain (Ancestors -> Active -> Descendants)
    /// Returns: The Card, The Frame it belongs to, and the Coordinate Conversion Scale
    /// The conversion scale is used to translate movement deltas between coordinate systems:
    ///   - Parent cards: scale < 1.0 (move slower - parent coords are smaller)
    ///   - Active cards: scale = 1.0 (normal movement)
    ///   - Child cards: scale > 1.0 (move faster - child coords are larger)
    func hitTestHierarchy(screenPoint: CGPoint,
                          viewSize: CGSize,
                          ignoringLocked: Bool = false) -> (card: Card, frame: Frame, conversionScale: Double, pointInFrame: SIMD2<Double>)? {

        // 1. Calculate Point in Active Frame (World Space)
        let pointActive = screenToWorldPixels_PureDouble(
            screenPoint,
            viewSize: viewSize,
            panOffset: panOffset,
            zoomScale: zoomScale,
            rotationAngle: rotationAngle
        )

        let transforms = collectFrameTransforms(pointActive: pointActive)

        // --- CHECK 1: DESCENDANTS (Foreground - Top Priority) ---
        for descendant in transforms.descendants.reversed() {
            for card in descendant.frame.cards.reversed() {
                if ignoringLocked, card.isLocked { continue }
                if card.hitTest(pointInFrame: descendant.point) {
                    return (card, descendant.frame, descendant.scale, descendant.point)
                }
            }
        }

        // --- CHECK 2: ACTIVE FRAME (Middle) ---
        for card in activeFrame.cards.reversed() {
            if ignoringLocked, card.isLocked { continue }
            if card.hitTest(pointInFrame: pointActive) {
                return (card, activeFrame, 1.0, pointActive)
            }
        }

        // --- CHECK 3: ANCESTORS (Background) ---
        for ancestor in transforms.ancestors {
            for card in ancestor.frame.cards.reversed() {
                if ignoringLocked, card.isLocked { continue }
                if card.hitTest(pointInFrame: ancestor.point) {
                    return (card, ancestor.frame, ancestor.scale, ancestor.point)
                }
            }
        }

        return nil
    }

    /// Handle long press gesture to open card settings
    /// Uses hierarchical hit testing to find cards at any depth level
    func handleLongPress(at point: CGPoint) {
        guard let view = metalView else { return }

        // Use hierarchical hit test to find card at any depth
        if let (card, _, _, _) = hitTestHierarchy(screenPoint: point, viewSize: view.bounds.size) {
            // Found a card! Notify SwiftUI
            onEditCard?(card)
        }
    }

    func deleteCard(_ card: Card) {
        guard let frame = findFrame(containing: card, in: rootFrame),
              let index = frame.cards.firstIndex(where: { $0 === card }) else { return }

        frame.cards.remove(at: index)
        card.isEditing = false
        clearLassoSelection()

        if case .card(let targetCard, _) = currentDrawingTarget, targetCard === card {
            currentDrawingTarget = nil
        }
    }

    private func findFrame(containing card: Card, in frame: Frame) -> Frame? {
        if frame.cards.contains(where: { $0 === card }) {
            return frame
        }
        for child in frame.children {
            if let found = findFrame(containing: card, in: child) {
                return found
            }
        }
        return nil
    }

    /// Add a new card to the canvas at the camera center
    /// The card will be a solid color and can be selected/dragged/edited
    /// Cards are created with constant screen size (300pt) regardless of zoom level
    func addCard() {
        guard let view = metalView else {
            return
        }

        // 1. Calculate camera center in world coordinates (where the user is looking)
        let cameraCenterWorld = calculateCameraCenterWorld(viewSize: view.bounds.size)

        // 2. Define Desired Screen Size (e.g., 300x200 points)
        // This ensures the card looks the same size to the user whether they are at 1x or 1000x zoom
        let screenWidth: Double = 300.0
        let screenHeight: Double = 200.0

        // 3. Convert to World Units
        // world = screen / zoom
        let worldW = screenWidth / zoomScale
        let worldH = screenHeight / zoomScale
        let cardSize = SIMD2<Double>(worldW, worldH)

        // 4. Use neon pink for high visibility against cyan background
        let neonPink = SIMD4<Float>(1.0, 0.0, 1.0, 1.0)  // Bright magenta

        // 5. Create the card (Default to Solid Color for now)
        // Capture current zoom so the card can correctly scale procedural backgrounds
        let card = Card(
            origin: cameraCenterWorld,
            size: cardSize,
            rotation: 0,
            zoom: zoomScale, // Capture current zoom!
            type: .solidColor(neonPink)
        )

        // 6. Add to active frame
        activeFrame.cards.append(card)
    }

    /// Create a stroke in a target frame's coordinate system (canvas strokes).
    /// Converts screen points into the target frame, even across telescope transitions.
    func createStrokeForFrame(screenPoints: [CGPoint],
                              frame: Frame,
                              viewSize: CGSize,
                              depthID: UInt32,
                              color: SIMD4<Float>? = nil,
                              depthWriteEnabled: Bool? = nil) -> Stroke {
        guard let transform = transformFromActive(to: frame) else {
            return Stroke(
                screenPoints: screenPoints,
                zoomAtCreation: zoomScale,
                panAtCreation: panOffset,
                viewSize: viewSize,
                rotationAngle: rotationAngle,
                color: color ?? brushSettings.color,
                baseWidth: brushSettings.size,
                zoomEffectiveAtCreation: Float(max(zoomScale, 1e-6)),
                device: device,
                depthID: depthID,
                depthWriteEnabled: depthWriteEnabled ?? brushSettings.depthWriteEnabled,
                constantScreenSize: brushSettings.constantScreenSize
            )
        }

        let effectiveZoom = max(zoomScale / transform.scale, 1e-6)
        var virtualScreenPoints: [CGPoint] = []
        virtualScreenPoints.reserveCapacity(screenPoints.count)

        for screenPt in screenPoints {
            let worldPtActive = screenToWorldPixels_PureDouble(
                screenPt,
                viewSize: viewSize,
                panOffset: panOffset,
                zoomScale: zoomScale,
                rotationAngle: rotationAngle
            )
            let targetWorld = worldPtActive * transform.scale + transform.translation
            virtualScreenPoints.append(CGPoint(x: targetWorld.x * effectiveZoom,
                                               y: targetWorld.y * effectiveZoom))
        }

        let finalColor = color ?? brushSettings.color
        let finalDepthWriteEnabled = depthWriteEnabled ?? brushSettings.depthWriteEnabled

        return Stroke(
            screenPoints: virtualScreenPoints,
            zoomAtCreation: effectiveZoom,
            panAtCreation: .zero,
            viewSize: .zero,
            rotationAngle: 0,
            color: finalColor,
            baseWidth: brushSettings.size,
            zoomEffectiveAtCreation: Float(effectiveZoom),
            device: device,
            depthID: depthID,
            depthWriteEnabled: finalDepthWriteEnabled,
            constantScreenSize: brushSettings.constantScreenSize
        )
    }

    /// Create a stroke in card-local coordinates
    /// Transforms screen-space points into the card's local coordinate system
    /// This ensures the stroke "sticks" to the card when it's moved or rotated
    ///
    /// **CROSS-DEPTH COMPATIBLE:**
    /// Supports drawing on cards anywhere in the telescope chain (ancestor/active/descendant).
    ///
    /// - Parameters:
    ///   - screenPoints: Raw screen-space touch points
    ///   - card: The card to draw on
    ///   - frame: The frame the card belongs to (may be parent, active, or child)
    ///   - viewSize: Screen dimensions
    /// - Returns: A stroke with points relative to card center
	    func createStrokeForCard(screenPoints: [CGPoint],
	                             card: Card,
	                             frame: Frame,
	                             viewSize: CGSize,
	                             depthID: UInt32,
	                             color: SIMD4<Float>? = nil,
	                             depthWriteEnabled: Bool? = nil) -> Stroke {
        // 1. Get the Card's World Position & Rotation (in its Frame)
        let cardOrigin = card.origin
        let cardRot = Double(card.rotation)

        // Pre-calculate rotation trig (inverse rotation to convert to card-local)
        let c = cos(-cardRot)
        let s = sin(-cardRot)

        guard let transform = transformFromActive(to: frame) else {
            return Stroke(
                screenPoints: screenPoints,
                zoomAtCreation: zoomScale,
                panAtCreation: panOffset,
                viewSize: viewSize,
                rotationAngle: rotationAngle,
                color: color ?? brushSettings.color,
                baseWidth: brushSettings.size,
                zoomEffectiveAtCreation: Float(max(zoomScale, 1e-6)),
                device: device,
                depthID: depthID,
                depthWriteEnabled: depthWriteEnabled ?? brushSettings.depthWriteEnabled,
                constantScreenSize: brushSettings.constantScreenSize
            )
        }

        // 2. Calculate Effective Zoom for the Card's Frame
        let effectiveZoom = max(zoomScale / transform.scale, 1e-6)

        // 3. Transform Screen Points -> Card Local Points
        var cardLocalPoints: [CGPoint] = []

        for screenPt in screenPoints {
            // A. Screen -> Active World (Standard conversion)
            let worldPtActive = screenToWorldPixels_PureDouble(
                screenPt,
                viewSize: viewSize,
                panOffset: panOffset,
                zoomScale: zoomScale,
                rotationAngle: rotationAngle
            )

            // B. Active World -> Card's Frame World (Apply telescope transform)
            let targetWorldPt = worldPtActive * transform.scale + transform.translation

            // C. Card's Frame World -> Card Local (Translate and Rotate)
            let dx = targetWorldPt.x - cardOrigin.x
            let dy = targetWorldPt.y - cardOrigin.y

            let localX = dx * c - dy * s
            let localY = dx * s + dy * c

            //  CRITICAL: Scale up to "virtual screen space"
            // We multiply by EFFECTIVE zoom so when Stroke.init divides by it,
            // we get back to world units (localX, localY) in the card's frame.
            let virtualScreenX = localX * effectiveZoom
            let virtualScreenY = localY * effectiveZoom

            cardLocalPoints.append(CGPoint(x: virtualScreenX, y: virtualScreenY))
        }

        let finalColor = color ?? brushSettings.color
        let finalDepthWriteEnabled = depthWriteEnabled ?? brushSettings.depthWriteEnabled

        // 5. Create the Stroke with Effective Zoom
	        return Stroke(
	            screenPoints: cardLocalPoints,   // Virtual screen space (world units * effectiveZoom)
	            zoomAtCreation: max(effectiveZoom, 1e-6),   // Use effective zoom for the card's frame!
	            panAtCreation: .zero,            // We handled position manually
	            viewSize: .zero,                 // We handled centering manually
	            rotationAngle: 0,                // We handled rotation manually
	            color: finalColor,               // Use brush settings color unless overridden
	            baseWidth: brushSettings.size,   // Use brush settings size
	            zoomEffectiveAtCreation: Float(max(effectiveZoom, 1e-6)),
	            device: device,                  // Pass device for buffer caching
	            depthID: depthID,
	            depthWriteEnabled: finalDepthWriteEnabled,
	            constantScreenSize: brushSettings.constantScreenSize
	        )
	    }

}

// MARK: - Gesture Delegate

extension TouchableMTKView: UIGestureRecognizerDelegate {
    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer,
                           shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer) -> Bool {
        true
    }
}
