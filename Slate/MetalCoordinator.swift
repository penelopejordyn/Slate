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
    var samplerState: MTLSamplerState!                  // Sampler for card textures
    var vertexBuffer: MTLBuffer!

    // Note: ICB removed - using simple GPU-offset approach instead

    //  Stencil States for Card Clipping
    var stencilStateDefault: MTLDepthStencilState! // Default passthrough (no testing)
    var stencilStateWrite: MTLDepthStencilState!   // Writes 1s to stencil (card background)
    var stencilStateRead: MTLDepthStencilState!    // Only draws where stencil == 1 (card strokes)
    var stencilStateClear: MTLDepthStencilState!   // Writes 0s to stencil (cleanup)

    // MARK: - Modal Input: Pencil vs. Finger

    /// Tracks what object we are currently drawing on with the pencil
    /// Now includes the Frame to support cross-depth drawing
    enum DrawingTarget {
        case canvas(Frame)
        case card(Card, Frame) // Track BOTH Card and the Frame it belongs to
    }
    var currentDrawingTarget: DrawingTarget?

    var currentTouchPoints: [CGPoint] = []      // Real historical points (SCREEN space)
    var predictedTouchPoints: [CGPoint] = []    // Future points (Transient, SCREEN space)
    var liveStrokeOrigin: SIMD2<Double>?        // Temporary origin for live stroke (Double precision)

    //  OPTIMIZATION: Adaptive Fidelity
    // Track last saved point for distance filtering to prevent vertex explosion during slow drawing
    var lastSavedPoint: CGPoint?

    // Telescoping Reference Frames
    // Instead of a flat array, we use a linked list of Frames for infinite zoom
    let rootFrame = Frame()           // The "Base Reality" - top level that cannot be zoomed out of
    lazy var activeFrame: Frame = rootFrame  // The current "Local Universe" we are viewing/editing

    weak var metalView: MTKView?

    // MARK: - Card Interaction Callbacks
    var onEditCard: ((Card) -> Void)?

    //  UPGRADED: Store camera state as Double for infinite precision
    var panOffset: SIMD2<Double> = .zero
    var zoomScale: Double = 1.0
    var rotationAngle: Float = 0.0

    // MARK: - Tile Rendering Controls

    /// Use baked tiles between these zoom ranges. Outside, fall back to raw strokes for fidelity.
    let tileZoomRange = 0.01...50.0

    // MARK: - Brush Settings
    let brushSettings = BrushSettings()

    // MARK: - Debug Stats
    var debugDrawnVerticesThisFrame: Int = 0
    var debugDrawnNodesThisFrame: Int = 0

    override init() {
        super.init()
        device = MTLCreateSystemDefaultDevice()!
        commandQueue = device.makeCommandQueue()!

        makePipeLine()
        makeVertexBuffer()
    }

    // MARK: - Commit 2: Recursive Renderer

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
                     excludedChild: Frame? = nil) { //  NEW: Prevent double-rendering

        let shouldUseTiles = shouldUseTileCache(for: currentZoom)

        // LAYER 1: RENDER PARENT (Background - Depth -1) -------------------------------
        if let parent = frame.parent {
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
                           excludedChild: frame) //  TELL PARENT TO SKIP US
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

        // Convert cullRadius from screen pixels to world units for inverse culling

        // DEBUG: Track culling stats
        var totalStrokes = 0
        var culledStrokes = 0

        if shouldUseTiles {
            drawTileLayer(frame: frame,
                          cameraCenter: cameraCenterInThisFrame,
                          viewSize: viewSize,
                          zoom: currentZoom,
                          rotation: currentRotation,
                          encoder: encoder)
        }

        // GPU-Offset Rendering: Direct draw with GPU applying offset
        encoder.setRenderPipelineState(pipelineState)

        for stroke in frame.strokes {
            if shouldUseTiles && isStrokeCoveredByTiles(stroke, in: frame) {
                continue // Tile already represents this stroke and is clean
            }

            guard !stroke.localVertices.isEmpty, let vertexBuffer = stroke.vertexBuffer else { continue }
            totalStrokes += 1

            let relativeOffsetDouble = stroke.origin - cameraCenterInThisFrame
            let ranges = stroke.visibleVertexRanges(relativeOffset: relativeOffsetDouble,
                                                   zoomScale: currentZoom,
                                                   cullRadius: cullRadius)
            if ranges.isEmpty {
                culledStrokes += 1
                continue // CULL!
            }

            debugDrawnVerticesThisFrame += stroke.localVertices.count
            debugDrawnNodesThisFrame += stroke.flatNodes.count

            let relativeOffset = SIMD2<Float>(
                Float(relativeOffsetDouble.x),
                Float(relativeOffsetDouble.y)
            )

            var transform = StrokeTransform(
                relativeOffset: relativeOffset,
                zoomScale: Float(currentZoom),
                screenWidth: Float(viewSize.width),
                screenHeight: Float(viewSize.height),
                rotationAngle: currentRotation,
                halfPixelWidth: Float(stroke.worldWidth * currentZoom * 0.5),
                vertexCount: UInt32(stroke.localVertices.count)
            )

            encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
            encoder.setVertexBytes(&transform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
            for range in ranges {
                encoder.drawPrimitives(type: stroke.primitiveType, vertexStart: range.0, vertexCount: range.1)
            }
        }

        // 2.2: RENDER CARDS (Middle layer - on top of canvas strokes)
        for card in frame.cards {
            // A. Calculate Position
            // Card lives in the Frame, so it moves with the Frame
            let relativeOffsetDouble = card.origin - cameraCenterInThisFrame

            // SCREEN SPACE CULLING for cards
            let distWorld = sqrt(relativeOffsetDouble.x * relativeOffsetDouble.x + relativeOffsetDouble.y * relativeOffsetDouble.y)
            let distScreen = distWorld * currentZoom
            let cardRadiusWorld = sqrt(pow(card.size.x, 2) + pow(card.size.y, 2)) * 0.5
            let cardRadiusScreen = cardRadiusWorld * currentZoom

            if (distScreen - cardRadiusScreen) > cullRadius {
                continue // Cull card
            }

            let relativeOffset = SIMD2<Float>(Float(relativeOffsetDouble.x), Float(relativeOffsetDouble.y))

            // B. Handle Rotation
            // Cards have their own rotation property
            // Total Rotation = Camera Rotation + Card Rotation
            let finalRotation = currentRotation + card.rotation

            var transform = CardTransform(
                relativeOffset: relativeOffset,
                zoomScale: Float(currentZoom),
                screenWidth: Float(viewSize.width),
                screenHeight: Float(viewSize.height),
                rotationAngle: finalRotation
            )

            //  STEP 1: DRAW CARD BACKGROUND + WRITE STENCIL
            // Write '1' into the stencil buffer where the card pixels are
            encoder.setDepthStencilState(stencilStateWrite)
            encoder.setStencilReferenceValue(1)

            // C. Set Pipeline & Bind Content Based on Card Type
            switch card.type {
            case .solidColor(let color):
                // Use solid color pipeline (no texture required)
                encoder.setRenderPipelineState(cardSolidPipelineState)
                encoder.setVertexBytes(&transform, length: MemoryLayout<CardTransform>.stride, index: 1)
                var c = color
                encoder.setFragmentBytes(&c, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)

            case .image(let texture):
                // Use textured pipeline (requires texture binding)
                encoder.setRenderPipelineState(cardPipelineState)
                encoder.setVertexBytes(&transform, length: MemoryLayout<CardTransform>.stride, index: 1)
                encoder.setFragmentTexture(texture, index: 0)
                encoder.setFragmentSamplerState(samplerState, index: 0)

            case .lined(let config):
                // Use procedural lined paper pipeline
                encoder.setRenderPipelineState(cardLinedPipelineState)
                encoder.setVertexBytes(&transform, length: MemoryLayout<CardTransform>.stride, index: 1)

                // 1. Background Color (Paper White)
                var bg = SIMD4<Float>(1, 1, 1, 1)
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

            case .grid(let config):
                // Use procedural grid paper pipeline
                encoder.setRenderPipelineState(cardGridPipelineState)
                encoder.setVertexBytes(&transform, length: MemoryLayout<CardTransform>.stride, index: 1)

                // 1. Background Color (Paper White)
                var bg = SIMD4<Float>(1, 1, 1, 1)
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
            if !card.strokes.isEmpty {
                encoder.setRenderPipelineState(pipelineState)
                encoder.setDepthStencilState(stencilStateRead) // <--- READ MODE
                encoder.setStencilReferenceValue(1)

                // Calculate "magic offset" for card-local coordinates
                let totalRotation = currentRotation + card.rotation
                let distX = card.origin.x - cameraCenterInThisFrame.x
                let distY = card.origin.y - cameraCenterInThisFrame.y
                let c = cos(Double(-card.rotation))
                let s = sin(Double(-card.rotation))
                let magicOffsetX = distX * c - distY * s
                let magicOffsetY = distX * s + distY * c
                let offset = SIMD2<Float>(Float(magicOffsetX), Float(magicOffsetY))

                // Draw card strokes directly (GPU applies offset)
                for stroke in card.strokes {
                    guard !stroke.localVertices.isEmpty, let vertexBuffer = stroke.vertexBuffer else { continue }
                    let strokeOffset = stroke.origin
                    let strokeRelativeOffsetDouble = SIMD2<Double>(Double(offset.x) + strokeOffset.x,
                                                                   Double(offset.y) + strokeOffset.y)

                    let ranges = stroke.visibleVertexRanges(relativeOffset: strokeRelativeOffsetDouble,
                                                           zoomScale: currentZoom,
                                                           cullRadius: cullRadius)

                    if ranges.isEmpty { continue }

                    let strokeRelativeOffset = SIMD2<Float>(Float(strokeRelativeOffsetDouble.x),
                                                            Float(strokeRelativeOffsetDouble.y))

                    debugDrawnVerticesThisFrame += stroke.localVertices.count
                    debugDrawnNodesThisFrame += stroke.flatNodes.count

                    var strokeTransform = StrokeTransform(
                        relativeOffset: strokeRelativeOffset,
                        zoomScale: Float(currentZoom),
                        screenWidth: Float(viewSize.width),
                        screenHeight: Float(viewSize.height),
                        rotationAngle: totalRotation,
                        halfPixelWidth: Float(stroke.worldWidth * currentZoom * 0.5),
                        vertexCount: UInt32(stroke.localVertices.count)
                    )

                    encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
                    encoder.setVertexBytes(&strokeTransform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
                    for range in ranges {
                        encoder.drawPrimitives(type: stroke.primitiveType, vertexStart: range.0, vertexCount: range.1)
                    }
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
            encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)

            // E. Draw Resize Handles (If Selected) 
            // Handles should not be affected by stencil
            if card.isEditing {
                encoder.setDepthStencilState(stencilStateDefault) // Disable stencil for handles
                drawCardHandles(card: card,
                                cameraCenter: cameraCenterInThisFrame,
                                viewSize: viewSize,
                                zoom: currentZoom,
                                rotation: finalRotation,
                                encoder: encoder)
            }
        }

        // Reset pipeline and stencil state for subsequent rendering (live stroke, child frames, etc.)
        encoder.setRenderPipelineState(pipelineState)
        encoder.setDepthStencilState(stencilStateDefault)

        // LAYER 3: RENDER CHILDREN (Foreground Details - Depth +1) ---------------------
        //  FIX: Look down into child frames so they don't disappear when zooming out
        for child in frame.children {
            //  FIX 1: Skip the excluded child (The one we came from)
            // This prevents "double vision" where the active frame renders itself twice:
            // once as a child of its parent, and once as the foreground layer.
            if let excluded = excludedChild, child === excluded {
                continue
            }

            // 1. Calculate effective zoom for the child
            // Child is smaller, so we zoom out (divide by scale)
            let childZoom = currentZoom / child.scaleRelativeToParent

            // Optimization: Culling
            // If child is too small to see (< 1 pixel equivalent), skip it
            if childZoom < 0.001 { continue }

            // 2. Calculate the Frame's offset relative to the camera (in Parent Units)
            let frameOffsetInParentUnits = child.originInParent - cameraCenterInThisFrame

            // 3. Convert that offset into Child Units
            // (Because the shader multiplies everything by childZoom, we must pre-scale the offset up)
            let frameOffsetInChildUnits = frameOffsetInParentUnits * child.scaleRelativeToParent

            // 4. Render child strokes directly (GPU applies offset)
            for stroke in child.strokes {
                guard !stroke.localVertices.isEmpty, let vertexBuffer = stroke.vertexBuffer else { continue }

                //  FIX: Add the Stroke's own origin to the Frame's offset
                let totalRelativeOffset = stroke.origin + frameOffsetInChildUnits
                let ranges = stroke.visibleVertexRanges(relativeOffset: totalRelativeOffset,
                                                        zoomScale: childZoom,
                                                        cullRadius: cullRadius)
                if ranges.isEmpty {
                    continue // Cull!
                }

                debugDrawnVerticesThisFrame += stroke.localVertices.count
                debugDrawnNodesThisFrame += stroke.flatNodes.count

                let childRelativeOffset = SIMD2<Float>(Float(totalRelativeOffset.x), Float(totalRelativeOffset.y))
                var childTransform = StrokeTransform(
                    relativeOffset: childRelativeOffset,
                    zoomScale: Float(childZoom),
                    screenWidth: Float(viewSize.width),
                    screenHeight: Float(viewSize.height),
                    rotationAngle: currentRotation,
                    halfPixelWidth: Float(stroke.worldWidth * childZoom * 0.5),
                    vertexCount: UInt32(stroke.localVertices.count)
                )

                encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
                encoder.setVertexBytes(&childTransform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
                for range in ranges {
                    encoder.drawPrimitives(type: stroke.primitiveType, vertexStart: range.0, vertexCount: range.1)
                }
            }

            // 5. Render Child Cards (Same logic as Layer 2, but with childZoom)
            for card in child.cards {
                // 1. Calculate Relative Position
                // Same logic as strokes: Frame Offset + Card Origin
                let totalRelativeOffset = card.origin + frameOffsetInChildUnits

                // SCREEN SPACE CULLING for child cards
                let distWorld = sqrt(totalRelativeOffset.x * totalRelativeOffset.x + totalRelativeOffset.y * totalRelativeOffset.y)
                let distScreen = distWorld * childZoom
                let cardRadiusWorld = sqrt(pow(card.size.x, 2) + pow(card.size.y, 2)) * 0.5
                let cardRadiusScreen = cardRadiusWorld * childZoom

                if (distScreen - cardRadiusScreen) > cullRadius {
                    continue // Cull child card
                }

                // 2. Setup Transform
                let relativeOffset = SIMD2<Float>(Float(totalRelativeOffset.x), Float(totalRelativeOffset.y))

                var childCardTransform = CardTransform(
                    relativeOffset: relativeOffset,
                    zoomScale: Float(childZoom), // Use the calculated childZoom
                    screenWidth: Float(viewSize.width),
                    screenHeight: Float(viewSize.height),
                    rotationAngle: currentRotation + card.rotation
                )

                // 3. Draw Card Background
                // Write to Stencil to clip strokes (same logic as Layer 2)
                encoder.setDepthStencilState(stencilStateWrite)
                encoder.setStencilReferenceValue(1)

                switch card.type {
                case .solidColor(let color):
                    encoder.setRenderPipelineState(cardSolidPipelineState)
                    encoder.setVertexBytes(&childCardTransform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
                    var c = color
                    encoder.setFragmentBytes(&c, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)

                case .image(let texture):
                    encoder.setRenderPipelineState(cardPipelineState)
                    encoder.setVertexBytes(&childCardTransform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
                    encoder.setFragmentTexture(texture, index: 0)
                    encoder.setFragmentSamplerState(samplerState, index: 0)

                case .lined(let config):
                    encoder.setRenderPipelineState(cardLinedPipelineState)
                    encoder.setVertexBytes(&childCardTransform, length: MemoryLayout<StrokeTransform>.stride, index: 1)

                    var bg = SIMD4<Float>(1, 1, 1, 1)
                    encoder.setFragmentBytes(&bg, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)

                    // FIX: Scale Spacing to World Units
                    let worldSpacing = config.spacing / Float(card.creationZoom)
                    let worldLineWidth = config.lineWidth / Float(card.creationZoom)

                    var uniforms = CardShaderUniforms(
                        spacing: worldSpacing,
                        lineWidth: worldLineWidth,
                        color: config.color,
                        cardWidth: Float(card.size.x),
                        cardHeight: Float(card.size.y)
                    )
                    encoder.setFragmentBytes(&uniforms, length: MemoryLayout<CardShaderUniforms>.stride, index: 1)

                case .grid(let config):
                    encoder.setRenderPipelineState(cardGridPipelineState)
                    encoder.setVertexBytes(&childCardTransform, length: MemoryLayout<StrokeTransform>.stride, index: 1)

                    var bg = SIMD4<Float>(1, 1, 1, 1)
                    encoder.setFragmentBytes(&bg, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)

                    // FIX: Scale Spacing to World Units
                    let worldSpacing = config.spacing / Float(card.creationZoom)
                    let worldLineWidth = config.lineWidth / Float(card.creationZoom)

                    var uniforms = CardShaderUniforms(
                        spacing: worldSpacing,
                        lineWidth: worldLineWidth,
                        color: config.color,
                        cardWidth: Float(card.size.x),
                        cardHeight: Float(card.size.y)
                    )
                    encoder.setFragmentBytes(&uniforms, length: MemoryLayout<CardShaderUniforms>.stride, index: 1)

                case .drawing: break // Handle later
                }

                // Bind Geometry & Draw
                let vBuffer = device.makeBuffer(bytes: card.localVertices, length: card.localVertices.count * MemoryLayout<StrokeVertex>.stride, options: [])
                encoder.setVertexBuffer(vBuffer, offset: 0, index: 0)
                encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)

                // 4. Render Card Strokes (Clipped) - Batched
                // FIX: Proper transform for child card strokes
                if !card.strokes.isEmpty {
                    encoder.setRenderPipelineState(pipelineState)
                    encoder.setDepthStencilState(stencilStateRead)
                    encoder.setStencilReferenceValue(1)

                    // Calculate total rotation (camera + card)
                    let totalRot = currentRotation + card.rotation

                    // Rotation matrix for card rotation
                    let c = cos(Double(card.rotation))
                    let s = sin(Double(card.rotation))

                    // Draw child card strokes directly (GPU applies offset)
                    for stroke in card.strokes {
                        guard !stroke.localVertices.isEmpty, let vertexBuffer = stroke.vertexBuffer else { continue }

                        debugDrawnVerticesThisFrame += stroke.localVertices.count
                        debugDrawnNodesThisFrame += stroke.flatNodes.count

                        // Transform Stroke Origin: Card Local -> Frame Local -> Camera Relative
                        let sx = stroke.origin.x
                        let sy = stroke.origin.y

                        let rotX = sx * c - sy * s
                        let rotY = sx * s + sy * c

                        let strokePos = totalRelativeOffset + SIMD2<Double>(rotX, rotY)
                        let ranges = stroke.visibleVertexRanges(relativeOffset: strokePos,
                                                                zoomScale: childZoom,
                                                                cullRadius: cullRadius)

                        if ranges.isEmpty { continue }

                        let strokeRelativeOffset = SIMD2<Float>(Float(strokePos.x), Float(strokePos.y))

                        var strokeTrans = StrokeTransform(
                            relativeOffset: strokeRelativeOffset,
                            zoomScale: Float(childZoom),
                            screenWidth: Float(viewSize.width),
                            screenHeight: Float(viewSize.height),
                            rotationAngle: totalRot,
                            halfPixelWidth: Float(stroke.worldWidth * childZoom * 0.5),
                            vertexCount: UInt32(stroke.localVertices.count)
                        )

                        encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
                        encoder.setVertexBytes(&strokeTrans, length: MemoryLayout<StrokeTransform>.stride, index: 1)
                        for range in ranges {
                            encoder.drawPrimitives(type: stroke.primitiveType, vertexStart: range.0, vertexCount: range.1)
                        }
                    }
                }

                // 5. Cleanup Stencil
                encoder.setRenderPipelineState(cardSolidPipelineState)
                encoder.setDepthStencilState(stencilStateClear)
                encoder.setStencilReferenceValue(0)
                encoder.setVertexBytes(&childCardTransform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
                var clearCol = SIMD4<Float>(0,0,0,0)
                encoder.setFragmentBytes(&clearCol, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
                encoder.setVertexBuffer(vBuffer, offset: 0, index: 0)
                encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
            }

            // Note: We do NOT recurse into grandchildren to avoid rendering depth ±2, ±3, etc.
            // Just immediate children (depth +1) is enough for visual continuity
        }
    }

    /// Determine if tiles should be used for the current zoom level.
    func shouldUseTileCache(for zoom: Double) -> Bool {
        return tileZoomRange.contains(zoom)
    }

    /// Calculate the visible world rect for the current camera and view size.
    func visibleWorldRect(cameraCenter: SIMD2<Double>, viewSize: CGSize, zoom: Double) -> CGRect {
        let halfWidth = Double(viewSize.width) * 0.5 / zoom
        let halfHeight = Double(viewSize.height) * 0.5 / zoom
        return CGRect(
            x: cameraCenter.x - halfWidth,
            y: cameraCenter.y - halfHeight,
            width: halfWidth * 2.0,
            height: halfHeight * 2.0
        )
    }

    /// Render baked tiles for the visible region of a frame.
    func drawTileLayer(frame: Frame,
                       cameraCenter: SIMD2<Double>,
                       viewSize: CGSize,
                       zoom: Double,
                       rotation: Float,
                       encoder: MTLRenderCommandEncoder) {
        guard shouldUseTileCache(for: zoom) else { return }

        let visibleRect = visibleWorldRect(cameraCenter: cameraCenter, viewSize: viewSize, zoom: zoom)
        let visibleKeys = frame.tileKeys(overlapping: visibleRect)

        guard !visibleKeys.isEmpty else { return }

        let halfW = Float(Frame.tileWorldSize / 2.0)
        let halfH = Float(Frame.tileWorldSize / 2.0)
        let white = SIMD4<Float>(1, 1, 1, 1)
        let quad: [StrokeVertex] = [
            StrokeVertex(position: SIMD2<Float>(-halfW, -halfH), uv: SIMD2<Float>(0, 1), color: white),
            StrokeVertex(position: SIMD2<Float>(-halfW,  halfH), uv: SIMD2<Float>(0, 0), color: white),
            StrokeVertex(position: SIMD2<Float>( halfW, -halfH), uv: SIMD2<Float>(1, 1), color: white),
            StrokeVertex(position: SIMD2<Float>(-halfW,  halfH), uv: SIMD2<Float>(0, 0), color: white),
            StrokeVertex(position: SIMD2<Float>( halfW,  halfH), uv: SIMD2<Float>(1, 0), color: white),
            StrokeVertex(position: SIMD2<Float>( halfW, -halfH), uv: SIMD2<Float>(1, 1), color: white)
        ]

        let quadBuffer = device.makeBuffer(bytes: quad,
                                           length: quad.count * MemoryLayout<StrokeVertex>.stride,
                                           options: .storageModeShared)

        encoder.setRenderPipelineState(cardPipelineState)
        encoder.setDepthStencilState(stencilStateDefault)

        for key in visibleKeys {
            guard let tile = frame.tiles[key], let texture = tile.texture, !tile.dirty else { continue }

            let center = CGPoint(x: tile.worldRect.midX, y: tile.worldRect.midY)
            let relativeOffset = SIMD2<Float>(
                Float(center.x - cameraCenter.x),
                Float(center.y - cameraCenter.y)
            )

            var transform = CardTransform(
                relativeOffset: relativeOffset,
                zoomScale: Float(zoom),
                screenWidth: Float(viewSize.width),
                screenHeight: Float(viewSize.height),
                rotationAngle: rotation
            )

            encoder.setVertexBuffer(quadBuffer, offset: 0, index: 0)
            encoder.setVertexBytes(&transform, length: MemoryLayout<CardTransform>.stride, index: 1)
            encoder.setFragmentTexture(texture, index: 0)
            encoder.setFragmentSamplerState(samplerState, index: 0)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: quad.count)
        }
    }

    /// Check if a stroke is fully covered by non-dirty tiles.
    func isStrokeCoveredByTiles(_ stroke: Stroke, in frame: Frame) -> Bool {
        let bounds = stroke.localBounds
        let rect = CGRect(
            x: stroke.origin.x + Double(bounds.origin.x),
            y: stroke.origin.y + Double(bounds.origin.y),
            width: Double(bounds.width),
            height: Double(bounds.height)
        )

        let keys = frame.tileKeys(overlapping: rect)
        guard !keys.isEmpty else { return false }

        for key in keys {
            guard let tile = frame.tiles[key], !tile.dirty, tile.texture != nil else {
                return false
            }
        }
        return true
    }

    /// Bake any dirty tiles on the frame into fixed-size textures.
    func bakeDirtyTiles(for frame: Frame, commandBuffer: MTLCommandBuffer) {
        let dirtyTiles = frame.tiles.values.filter { $0.dirty }
        guard !dirtyTiles.isEmpty else { return }

        let tileZoom = Double(Frame.tileTextureSize) / Frame.tileWorldSize
        let clearColor = metalView?.clearColor ?? MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

        for var tile in dirtyTiles {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .bgra8Unorm,
                width: Frame.tileTextureSize,
                height: Frame.tileTextureSize,
                mipmapped: false
            )
            descriptor.usage = [.renderTarget, .shaderRead]

            if tile.texture == nil || tile.texture?.width != Frame.tileTextureSize {
                tile.texture = device.makeTexture(descriptor: descriptor)
            }
            guard let texture = tile.texture else { continue }

            let rpd = MTLRenderPassDescriptor()
            rpd.colorAttachments[0].texture = texture
            rpd.colorAttachments[0].loadAction = .clear
            rpd.colorAttachments[0].storeAction = .store
            rpd.colorAttachments[0].clearColor = clearColor

            guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: rpd) else { continue }
            encoder.setRenderPipelineState(pipelineState)
            encoder.setViewport(MTLViewport(originX: 0,
                                            originY: 0,
                                            width: Double(Frame.tileTextureSize),
                                            height: Double(Frame.tileTextureSize),
                                            znear: 0,
                                            zfar: 1))

            let tileCenter = SIMD2<Double>(tile.worldRect.midX, tile.worldRect.midY)

            for stroke in frame.strokes {
                let bounds = stroke.localBounds
                let rect = CGRect(
                    x: stroke.origin.x + Double(bounds.origin.x),
                    y: stroke.origin.y + Double(bounds.origin.y),
                    width: Double(bounds.width),
                    height: Double(bounds.height)
                )

                guard rect.intersects(tile.worldRect) else { continue }
                guard let vertexBuffer = stroke.vertexBuffer, !stroke.localVertices.isEmpty else { continue }

                let relativeOffsetDouble = stroke.origin - tileCenter
                let relativeOffset = SIMD2<Float>(Float(relativeOffsetDouble.x), Float(relativeOffsetDouble.y))

                var transform = StrokeTransform(
                    relativeOffset: relativeOffset,
                    zoomScale: Float(tileZoom),
                    screenWidth: Float(Frame.tileTextureSize),
                    screenHeight: Float(Frame.tileTextureSize),
                    rotationAngle: 0
                )

                encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
                encoder.setVertexBytes(&transform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
                encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: stroke.localVertices.count)
            }

            encoder.endEncoding()
            tile.dirty = false
            frame.tiles[tile.key] = tile
        }
    }

    /// Bake tiles in the active hierarchy before presenting the drawable.
    func bakeVisibleTiles(commandBuffer: MTLCommandBuffer, viewSize: CGSize) {
        guard shouldUseTileCache(for: zoomScale) else { return }
        _ = viewSize // Reserved for future viewport-aware baking heuristics

        var framesToBake: [Frame] = [activeFrame]
        if let parent = activeFrame.parent {
            framesToBake.append(parent)
        }
        framesToBake.append(contentsOf: activeFrame.children)

        for frame in framesToBake {
            bakeDirtyTiles(for: frame, commandBuffer: commandBuffer)
        }
    }

    // Note: ICB encoding functions removed - using simple GPU-offset rendering

    ///  CONSTANT SCREEN SIZE HANDLES
    /// Draws resize handles at card corners that maintain 12pt screen size regardless of zoom
    /// This provides consistent UX across all zoom levels (infinite canvas requirement)
    ///
    /// - Parameters:
    ///   - card: The card to draw handles for
    ///   - cameraCenter: Camera position in frame coordinates
    ///   - viewSize: Screen dimensions
    ///   - zoom: Current zoom level
    ///   - rotation: Total rotation (camera + card)
    ///   - encoder: Metal render encoder
    func drawCardHandles(card: Card,
                         cameraCenter: SIMD2<Double>,
                         viewSize: CGSize,
                         zoom: Double,
                         rotation: Float,
                         encoder: MTLRenderCommandEncoder) {

        // 1. Calculate Constant Screen Size
        // We want handles to be ~12pt on screen regardless of zoom
        // In World Space, that is: 12.0 / Zoom
        let handleSizeScreen: Float = 12.0
        let handleSizeWorld = handleSizeScreen / Float(zoom)

        // 2. Generate Handle Geometry (A small quad centered at 0,0)
        // We generate this on the fly because the size changes every frame (based on zoom)
        let halfS = handleSizeWorld / 2.0
        let handleColor = SIMD4<Float>(0.0, 0.47, 1.0, 1.0) // Blue handle color
        let baseHandleVertices: [SIMD2<Float>] = [
            SIMD2<Float>(-halfS, -halfS), SIMD2<Float>(-halfS,  halfS), SIMD2<Float>( halfS, -halfS),
            SIMD2<Float>(-halfS,  halfS), SIMD2<Float>( halfS, -halfS), SIMD2<Float>( halfS,  halfS)
        ]

        // 3. Calculate Corner Positions
        // We need to place these squares at the card's corners
        // IMPORTANT: The handles must rotate WITH the card
        let corners = card.getLocalCorners() // TL, TR, BL, BR

        // NOTE: Base transform not needed - each handle gets its own transform with relativeOffset

        // 5. Draw Loop - One handle at each corner
        for cornerLocal in corners {
            // Calculate the Handle's World Origin on CPU
            let cardRot = card.rotation
            let c = cos(cardRot)
            let s = sin(cardRot)

            // Rotate the corner offset by card rotation
            let rotatedCornerX = Double(cornerLocal.x) * Double(c) - Double(cornerLocal.y) * Double(s)
            let rotatedCornerY = Double(cornerLocal.x) * Double(s) + Double(cornerLocal.y) * Double(c)

            // Absolute World Position of this handle
            let handleOriginX = card.origin.x + rotatedCornerX
            let handleOriginY = card.origin.y + rotatedCornerY

            // Calculate Relative Offset (camera-relative position)
            let relativeOffset = SIMD2<Float>(Float(handleOriginX - cameraCenter.x),
                                              Float(handleOriginY - cameraCenter.y))

            // Use GPU offset approach
            let handleVertices = baseHandleVertices.map { pos in
                StrokeVertex(position: pos, uv: .zero, color: handleColor)
            }

            var handleTransform = StrokeTransform(
                relativeOffset: relativeOffset,
                zoomScale: Float(zoom),
                screenWidth: Float(viewSize.width),
                screenHeight: Float(viewSize.height),
                rotationAngle: rotation,
                halfPixelWidth: 0,
                vertexCount: UInt32(handleVertices.count)
            )

            // Create buffer and draw
            let vertexBuffer = device.makeBuffer(bytes: handleVertices,
                                                 length: handleVertices.count * MemoryLayout<StrokeVertex>.stride,
                                                 options: [])
            encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
            encoder.setVertexBytes(&handleTransform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
        }
    }

    func draw(in view: MTKView) {
        // Reset per-frame debug counters
        debugDrawnVerticesThisFrame = 0
        debugDrawnNodesThisFrame = 0

        // Calculate Camera Center in World Space (Double precision)
        // This is the "View Center" - where the center of the screen is in the infinite world.
        let cameraCenterWorld = calculateCameraCenterWorld(viewSize: view.bounds.size)

        // Start rendering pipeline
        let commandBuffer = commandQueue.makeCommandBuffer()!

        // Bake dirty tiles in the background before drawing the frame
        bakeVisibleTiles(commandBuffer: commandBuffer, viewSize: view.bounds.size)

        guard let rpd = view.currentRenderPassDescriptor else { return }
        let enc = commandBuffer.makeRenderCommandEncoder(descriptor: rpd)!
        enc.setRenderPipelineState(pipelineState)
        enc.setCullMode(.none)

        // Recursive Rendering
        // Render all committed strokes using the recursive renderer
        // This will automatically render parent frames (background) before the active frame (foreground)
        renderFrame(activeFrame,
                   cameraCenterInThisFrame: cameraCenterWorld,
                   viewSize: view.bounds.size,
                   currentZoom: zoomScale,
                   currentRotation: rotationAngle,
                   encoder: enc,
                   excludedChild: nil) // Start with no exclusion

        // Live Stroke Rendering
        // Live strokes are rendered on top of all committed strokes (foreground)
        // Combine real + predicted points for latency reduction
        if (currentTouchPoints.count + predictedTouchPoints.count) >= 2, let tempOrigin = liveStrokeOrigin {
            let zoom = Double(zoomScale)

            // Determine effective zoom and stroke width for cross-depth drawing
            var effectiveZoom: Double = zoom
            var worldWidth: Double

            if case .card(_, let frame) = currentDrawingTarget {
                // Calculate effective zoom for the card's frame
                if frame === activeFrame {
                    // Card is in active frame - use active zoom
                    effectiveZoom = zoom
                } else if let p = activeFrame.parent, frame === p {
                    // Card is in parent frame - parent is zoomed IN from our perspective
                    // effectiveZoom = activeZoom * activeScale (we see parent coords magnified)
                    effectiveZoom = zoom * activeFrame.scaleRelativeToParent
                } else if let child = activeFrame.children.first(where: { $0 === frame }) {
                    // Card is in child frame - child is zoomed OUT from our perspective
                    // effectiveZoom = activeZoom / childScale (we see child coords shrunk)
                    effectiveZoom = zoom / child.scaleRelativeToParent
                }

                // Width should be based on effective zoom for the card's frame
                worldWidth = brushSettings.size / effectiveZoom
            } else {
                // Drawing on canvas - normal calculation
                worldWidth = brushSettings.size / zoom
            }

            //  COMBINE REAL + PREDICTED
            // We create a temporary array just for this frame's rendering
            let pointsToDraw = currentTouchPoints + predictedTouchPoints

            var localPoints: [SIMD2<Float>]
            var liveTransform: StrokeTransform
            var liveRelativeOffset: SIMD2<Float>
            var strokeZoomForWidth: Double = zoom

            //  Handle card vs canvas drawing differently
            if case .card(let card, let frame) = currentDrawingTarget {
                // CARD DRAWING (Cross-Depth Compatible): Transform to card-local coordinates
                let firstScreenPt = currentTouchPoints[0]  // Origin is always the REAL first point
                let cardOrigin = card.origin
                let cardRot = Double(card.rotation)
                let cameraRot = Double(rotationAngle)

                // Determine Scale Factor (Active -> Card Frame)
                var frameScale: Double = 1.0
                if frame === activeFrame {
                    frameScale = 1.0
                } else if let p = activeFrame.parent, frame === p {
                    frameScale = 1.0 / activeFrame.scaleRelativeToParent
                } else if activeFrame.children.contains(where: { $0 === frame }) {
                    frameScale = frame.scaleRelativeToParent
                }

                // Pre-calculate card inverse rotation
                let cc = cos(-cardRot)
                let ss = sin(-cardRot)

                // Pre-calculate camera inverse rotation
                let cCam = cos(cameraRot)
                let sCam = sin(cameraRot)

                localPoints = pointsToDraw.map { pt in
                    // A. Screen → Active World (un-rotate by camera, apply zoom)
                    let dx = Double(pt.x) - Double(firstScreenPt.x)
                    let dy = Double(pt.y) - Double(firstScreenPt.y)

                    let unrotatedX = dx * cCam + dy * sCam
                    let unrotatedY = -dx * sCam + dy * cCam

                    let worldDx = unrotatedX / zoom
                    let worldDy = unrotatedY / zoom

                    // B. Add to first touch world position (in Active Frame)
                    let firstWorldPt = screenToWorldPixels_PureDouble(
                        firstScreenPt,
                        viewSize: view.bounds.size,
                        panOffset: panOffset,
                        zoomScale: zoomScale,
                        rotationAngle: rotationAngle
                    )
                    let worldX = firstWorldPt.x + worldDx
                    let worldY = firstWorldPt.y + worldDy

                    // C. Active World → Card's Frame World (Apply scale transform)
                    var targetWorldPt = SIMD2<Double>(worldX, worldY)
                    if frame !== activeFrame {
                        if frameScale > 1.0 {
                            // Child Frame: Translate and Scale UP
                            targetWorldPt = (SIMD2<Double>(worldX, worldY) - frame.originInParent) * frame.scaleRelativeToParent
                        } else {
                            // Parent Frame: Scale DOWN and Translate
                            targetWorldPt = activeFrame.originInParent + (SIMD2<Double>(worldX, worldY) / activeFrame.scaleRelativeToParent)
                        }
                    }

                    // D. Card's Frame World → Card Local (subtract card origin, un-rotate by card)
                    let cardDx = targetWorldPt.x - cardOrigin.x
                    let cardDy = targetWorldPt.y - cardOrigin.y

                    let localX = cardDx * cc - cardDy * ss
                    let localY = cardDx * ss + cardDy * cc

                    return SIMD2<Float>(Float(localX), Float(localY))
                }

                // Calculate transform for cross-depth rendering
                // Key insight: localPoints are in the card's frame coordinate system
                // We need to render them with the appropriate zoom for that frame
                var magicOffsetX: Double
                var magicOffsetY: Double
                var renderZoom: Double

                if frame === activeFrame {
                    // Card is in active frame - use standard approach
                    let cardRelativeOffset = card.origin - cameraCenterWorld
                    let c = cos(Double(-card.rotation))
                    let s = sin(Double(-card.rotation))
                    magicOffsetX = cardRelativeOffset.x * c - cardRelativeOffset.y * s
                    magicOffsetY = cardRelativeOffset.x * s + cardRelativeOffset.y * c
                    renderZoom = zoomScale

                } else if let p = activeFrame.parent, frame === p {
                    // Card is in parent frame
                    // Convert camera position to parent frame coordinates
                    let cameraCenterInParent = activeFrame.originInParent + (cameraCenterWorld / activeFrame.scaleRelativeToParent)
                    let cardRelativeOffset = card.origin - cameraCenterInParent
                    let c = cos(Double(-card.rotation))
                    let s = sin(Double(-card.rotation))
                    magicOffsetX = cardRelativeOffset.x * c - cardRelativeOffset.y * s
                    magicOffsetY = cardRelativeOffset.x * s + cardRelativeOffset.y * c

                    // Parent is zoomed IN from our perspective (we see it magnified)
                    // Use the effective zoom for parent coords
                    renderZoom = effectiveZoom

                } else if let child = activeFrame.children.first(where: { $0 === frame }) {
                    // Card is in child frame
                    // Convert camera position to child frame coordinates
                    let cameraCenterInChild = (cameraCenterWorld - child.originInParent) * child.scaleRelativeToParent
                    let cardRelativeOffset = card.origin - cameraCenterInChild
                    let c = cos(Double(-card.rotation))
                    let s = sin(Double(-card.rotation))
                    magicOffsetX = cardRelativeOffset.x * c - cardRelativeOffset.y * s
                    magicOffsetY = cardRelativeOffset.x * s + cardRelativeOffset.y * c

                    // Child is zoomed OUT from our perspective (we see it shrunk)
                    // Use the effective zoom for child coords
                    renderZoom = effectiveZoom

                } else {
                    // Fallback
                    let cardRelativeOffset = card.origin - cameraCenterWorld
                    let c = cos(Double(-card.rotation))
                    let s = sin(Double(-card.rotation))
                    magicOffsetX = cardRelativeOffset.x * c - cardRelativeOffset.y * s
                    magicOffsetY = cardRelativeOffset.x * s + cardRelativeOffset.y * c
                    renderZoom = zoomScale
                }

                liveRelativeOffset = SIMD2<Float>(Float(magicOffsetX), Float(magicOffsetY))
                liveTransform = StrokeTransform(
                    relativeOffset: liveRelativeOffset,
                    zoomScale: Float(renderZoom),  // Use effective zoom for the card's frame!
                    screenWidth: Float(view.bounds.width),
                    screenHeight: Float(view.bounds.height),
                    rotationAngle: rotationAngle + card.rotation,
                    halfPixelWidth: 0,
                    vertexCount: 0
                )
                strokeZoomForWidth = renderZoom
            } else {
                // CANVAS DRAWING: Use original approach
                let firstScreenPt = currentTouchPoints[0]  // Origin is always the REAL first point
                let angle = Double(rotationAngle)
                let c = cos(angle)
                let s = sin(angle)

                localPoints = pointsToDraw.map { pt in
                    let dx = Double(pt.x) - Double(firstScreenPt.x)
                    let dy = Double(pt.y) - Double(firstScreenPt.y)

                    let unrotatedX = dx * c + dy * s
                    let unrotatedY = -dx * s + dy * c

                    let worldDx = unrotatedX / zoom
                    let worldDy = unrotatedY / zoom

                    return SIMD2<Float>(Float(worldDx), Float(worldDy))
                }

                let relativeOffsetDouble = tempOrigin - cameraCenterWorld
                liveRelativeOffset = SIMD2<Float>(Float(relativeOffsetDouble.x),
                                                   Float(relativeOffsetDouble.y))
                liveTransform = StrokeTransform(
                    relativeOffset: liveRelativeOffset,
                    zoomScale: Float(zoomScale),
                    screenWidth: Float(view.bounds.width),
                    screenHeight: Float(view.bounds.height),
                    rotationAngle: rotationAngle,
                    halfPixelWidth: 0,
                    vertexCount: 0
                )
                strokeZoomForWidth = zoomScale
            }

            // Tessellate in LOCAL space
            let liveStrokeVertices = buildStrokeStripVertices(
                centerPoints: localPoints,
                width: Float(worldWidth),
                color: brushSettings.color
            )

            guard !liveStrokeVertices.isEmpty else {
                enc.endEncoding()
                commandBuffer.present(view.currentDrawable!)
                commandBuffer.commit()
                return
            }

            liveTransform.halfPixelWidth = Float(worldWidth * strokeZoomForWidth * 0.5)
            liveTransform.vertexCount = UInt32(localVertices.count)

            //  If drawing on a card, set up stencil clipping for the live preview
            if case .card(let card, let frame) = currentDrawingTarget {
                // STEP 1: Write card's stencil mask
                enc.setDepthStencilState(stencilStateWrite)
                enc.setStencilReferenceValue(1)
                enc.setRenderPipelineState(cardSolidPipelineState)

                // Calculate card transform - MUST match live stroke transform!
                // Use the same coordinate system and zoom as the live stroke
                var stencilOffsetX: Double
                var stencilOffsetY: Double
                var stencilZoom: Double

                if frame === activeFrame {
                    // Card is in active frame
                    let cardRelativeOffset = card.origin - cameraCenterWorld
                    let c = cos(Double(-card.rotation))
                    let s = sin(Double(-card.rotation))
                    stencilOffsetX = cardRelativeOffset.x * c - cardRelativeOffset.y * s
                    stencilOffsetY = cardRelativeOffset.x * s + cardRelativeOffset.y * c
                    stencilZoom = zoomScale

                } else if let p = activeFrame.parent, frame === p {
                    // Card is in parent frame
                    let cameraCenterInParent = activeFrame.originInParent + (cameraCenterWorld / activeFrame.scaleRelativeToParent)
                    let cardRelativeOffset = card.origin - cameraCenterInParent
                    let c = cos(Double(-card.rotation))
                    let s = sin(Double(-card.rotation))
                    stencilOffsetX = cardRelativeOffset.x * c - cardRelativeOffset.y * s
                    stencilOffsetY = cardRelativeOffset.x * s + cardRelativeOffset.y * c
                    stencilZoom = effectiveZoom

                } else if let child = activeFrame.children.first(where: { $0 === frame }) {
                    // Card is in child frame
                    let cameraCenterInChild = (cameraCenterWorld - child.originInParent) * child.scaleRelativeToParent
                    let cardRelativeOffset = card.origin - cameraCenterInChild
                    let c = cos(Double(-card.rotation))
                    let s = sin(Double(-card.rotation))
                    stencilOffsetX = cardRelativeOffset.x * c - cardRelativeOffset.y * s
                    stencilOffsetY = cardRelativeOffset.x * s + cardRelativeOffset.y * c
                    stencilZoom = effectiveZoom

                } else {
                    // Fallback
                    let cardRelativeOffset = card.origin - cameraCenterWorld
                    let c = cos(Double(-card.rotation))
                    let s = sin(Double(-card.rotation))
                    stencilOffsetX = cardRelativeOffset.x * c - cardRelativeOffset.y * s
                    stencilOffsetY = cardRelativeOffset.x * s + cardRelativeOffset.y * c
                    stencilZoom = zoomScale
                }

                let cardRotation = rotationAngle + card.rotation
                var cardTransform = CardTransform(
                    relativeOffset: SIMD2<Float>(Float(stencilOffsetX), Float(stencilOffsetY)),
                    zoomScale: Float(stencilZoom),  // Use effective zoom to match live stroke!
                    screenWidth: Float(view.bounds.width),
                    screenHeight: Float(view.bounds.height),
                    rotationAngle: cardRotation
                )

                // Draw card quad to stencil buffer
                enc.setVertexBytes(&cardTransform, length: MemoryLayout<CardTransform>.stride, index: 1)
                var clearColor = SIMD4<Float>(0, 0, 0, 0) // Transparent (won't affect color buffer)
                enc.setFragmentBytes(&clearColor, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)

                let cardVertexBuffer = device.makeBuffer(
                    bytes: card.localVertices,
                    length: card.localVertices.count * MemoryLayout<StrokeVertex>.stride,
                    options: .storageModeShared
                )
                enc.setVertexBuffer(cardVertexBuffer, offset: 0, index: 0)
                enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)

                // STEP 2: Set stencil read mode for live stroke (clip to card)
                enc.setDepthStencilState(stencilStateRead)
                enc.setStencilReferenceValue(1)
                enc.setRenderPipelineState(pipelineState)
            } else {
                // Drawing on canvas - reset to stroke pipeline
                // FIX: The recursive renderer may have left cardSolidPipelineState active
                // after rendering child frame cards. We must explicitly reset to pipelineState.
                enc.setRenderPipelineState(pipelineState)
                enc.setDepthStencilState(stencilStateDefault)
            }

            // Create buffers and render live stroke (GPU applies offset)
            let livePrimitive: MTLPrimitiveType = localPoints.count > 1 ? .triangleStrip : .triangle

            let vertexBuffer = device.makeBuffer(
                bytes: localVertices,
                length: localVertices.count * MemoryLayout<StrokeVertex>.stride,
                options: .storageModeShared
            )

            enc.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
            enc.setVertexBytes(&liveTransform, length: MemoryLayout<StrokeTransform>.stride, index: 1)

            debugDrawnVerticesThisFrame += liveStrokeVertices.count

            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: liveStrokeVertices.count)

            // STEP 3: Clean up stencil if we used it
            if case .card(let card, let frame) = currentDrawingTarget {
                enc.setDepthStencilState(stencilStateClear)
                enc.setStencilReferenceValue(0)
                enc.setRenderPipelineState(cardSolidPipelineState)

                // Calculate card transform - MUST match stencil write transform!
                var stencilOffsetX: Double
                var stencilOffsetY: Double
                var stencilZoom: Double

                if frame === activeFrame {
                    let cardRelativeOffset = card.origin - cameraCenterWorld
                    let c = cos(Double(-card.rotation))
                    let s = sin(Double(-card.rotation))
                    stencilOffsetX = cardRelativeOffset.x * c - cardRelativeOffset.y * s
                    stencilOffsetY = cardRelativeOffset.x * s + cardRelativeOffset.y * c
                    stencilZoom = zoomScale

                } else if let p = activeFrame.parent, frame === p {
                    let cameraCenterInParent = activeFrame.originInParent + (cameraCenterWorld / activeFrame.scaleRelativeToParent)
                    let cardRelativeOffset = card.origin - cameraCenterInParent
                    let c = cos(Double(-card.rotation))
                    let s = sin(Double(-card.rotation))
                    stencilOffsetX = cardRelativeOffset.x * c - cardRelativeOffset.y * s
                    stencilOffsetY = cardRelativeOffset.x * s + cardRelativeOffset.y * c
                    stencilZoom = effectiveZoom

                } else if let child = activeFrame.children.first(where: { $0 === frame }) {
                    let cameraCenterInChild = (cameraCenterWorld - child.originInParent) * child.scaleRelativeToParent
                    let cardRelativeOffset = card.origin - cameraCenterInChild
                    let c = cos(Double(-card.rotation))
                    let s = sin(Double(-card.rotation))
                    stencilOffsetX = cardRelativeOffset.x * c - cardRelativeOffset.y * s
                    stencilOffsetY = cardRelativeOffset.x * s + cardRelativeOffset.y * c
                    stencilZoom = effectiveZoom

                } else {
                    let cardRelativeOffset = card.origin - cameraCenterWorld
                    let c = cos(Double(-card.rotation))
                    let s = sin(Double(-card.rotation))
                    stencilOffsetX = cardRelativeOffset.x * c - cardRelativeOffset.y * s
                    stencilOffsetY = cardRelativeOffset.x * s + cardRelativeOffset.y * c
                    stencilZoom = zoomScale
                }

                // Redraw card quad to clear stencil
                var cardTransform = CardTransform(
                    relativeOffset: SIMD2<Float>(Float(stencilOffsetX), Float(stencilOffsetY)),
                    zoomScale: Float(stencilZoom),
                    screenWidth: Float(view.bounds.width),
                    screenHeight: Float(view.bounds.height),
                    rotationAngle: rotationAngle + card.rotation
                )
                enc.setVertexBytes(&cardTransform, length: MemoryLayout<CardTransform>.stride, index: 1)
                var clearColor = SIMD4<Float>(0, 0, 0, 0)
                enc.setFragmentBytes(&clearColor, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)

                let cardVertexBuffer = device.makeBuffer(
                    bytes: card.localVertices,
                    length: card.localVertices.count * MemoryLayout<StrokeVertex>.stride,
                    options: .storageModeShared
                )
                enc.setVertexBuffer(cardVertexBuffer, offset: 0, index: 0)
                enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)

                // Reset to default
                enc.setDepthStencilState(stencilStateDefault)
            }
        }

        enc.endEncoding()
        commandBuffer.present(view.currentDrawable!)
        commandBuffer.commit()

        // Update Debug HUD
        updateDebugHUD(view: view)
    }

    /// Update the debug HUD with current frame depth and zoom level
    func updateDebugHUD(view: MTKView) {
        // Access debugLabel through the stored metalView reference
        guard let mtkView = metalView else { return }

        // Find the debug label subview
        guard let debugLabel = mtkView.subviews.compactMap({ $0 as? UILabel }).first else { return }

        // Calculate frame depth
        var depth = 0
        var current: Frame? = activeFrame
        while current?.parent != nil {
            depth += 1
            current = current?.parent
        }

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
        } else {
            effectiveText = String(format: "%.1f×", effectiveZoom)
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
            Debug: Verts \(self.debugDrawnVerticesThisFrame) | Nodes \(self.debugDrawnNodesThisFrame)
            Camera: \(cameraPosText)
            """
        }
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func makePipeLine() {
        let library = device.makeDefaultLibrary()!

        // Stroke Pipeline (Batched Rendering with Color per Vertex)
        let strokeVertexDesc = MTLVertexDescriptor()
        // Position attribute (attribute 0)
        strokeVertexDesc.attributes[0].format = .float2
        strokeVertexDesc.attributes[0].offset = 0
        strokeVertexDesc.attributes[0].bufferIndex = 0
        // UV attribute (attribute 1)
        strokeVertexDesc.attributes[1].format = .float2
        strokeVertexDesc.attributes[1].offset = MemoryLayout<SIMD2<Float>>.stride
        strokeVertexDesc.attributes[1].bufferIndex = 0
        // Color attribute (attribute 2) - NEW for batching
        strokeVertexDesc.attributes[2].format = .float4
        strokeVertexDesc.attributes[2].offset = MemoryLayout<SIMD2<Float>>.stride * 2
        strokeVertexDesc.attributes[2].bufferIndex = 0
        // Layout
        strokeVertexDesc.layouts[0].stride = MemoryLayout<StrokeVertex>.stride
        strokeVertexDesc.layouts[0].stepFunction = .perVertex

        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction   = library.makeFunction(name: "vertex_main")
        desc.fragmentFunction = library.makeFunction(name: "fragment_main")
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        desc.vertexDescriptor = strokeVertexDesc
        desc.stencilAttachmentPixelFormat = .stencil8 //  Required for stencil buffer
        pipelineState = try? device.makeRenderPipelineState(descriptor: desc)

        // Setup vertex descriptor for StrokeVertex structure (shared by both card pipelines)
        let vertexDesc = MTLVertexDescriptor()
        // Position attribute (attribute 0)
        vertexDesc.attributes[0].format = .float2
        vertexDesc.attributes[0].offset = 0
        vertexDesc.attributes[0].bufferIndex = 0
        // UV attribute (attribute 1)
        vertexDesc.attributes[1].format = .float2
        vertexDesc.attributes[1].offset = MemoryLayout<SIMD2<Float>>.stride
        vertexDesc.attributes[1].bufferIndex = 0
        // Layout
        vertexDesc.layouts[0].stride = MemoryLayout<StrokeVertex>.stride
        vertexDesc.layouts[0].stepFunction = .perVertex

        // Textured Card Pipeline (for images, PDFs)
        let cardDesc = MTLRenderPipelineDescriptor()
        cardDesc.vertexFunction   = library.makeFunction(name: "vertex_card")
        cardDesc.fragmentFunction = library.makeFunction(name: "fragment_card_texture")
        cardDesc.colorAttachments[0].pixelFormat = .bgra8Unorm
        cardDesc.colorAttachments[0].isBlendingEnabled = true
        cardDesc.colorAttachments[0].rgbBlendOperation = .add
        cardDesc.colorAttachments[0].alphaBlendOperation = .add
        cardDesc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        cardDesc.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        cardDesc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        cardDesc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        cardDesc.vertexDescriptor = vertexDesc
        cardDesc.stencilAttachmentPixelFormat = .stencil8 //  Required for stencil buffer
        cardPipelineState = try? device.makeRenderPipelineState(descriptor: cardDesc)

        // Solid Color Card Pipeline (for placeholders, backgrounds)
        let cardSolidDesc = MTLRenderPipelineDescriptor()
        cardSolidDesc.vertexFunction   = library.makeFunction(name: "vertex_card")
        cardSolidDesc.fragmentFunction = library.makeFunction(name: "fragment_card_solid")
        cardSolidDesc.colorAttachments[0].pixelFormat = .bgra8Unorm
        cardSolidDesc.colorAttachments[0].isBlendingEnabled = true
        cardSolidDesc.colorAttachments[0].rgbBlendOperation = .add
        cardSolidDesc.colorAttachments[0].alphaBlendOperation = .add
        cardSolidDesc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        cardSolidDesc.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        cardSolidDesc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        cardSolidDesc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        cardSolidDesc.vertexDescriptor = vertexDesc
        cardSolidDesc.stencilAttachmentPixelFormat = .stencil8 //  Required for stencil buffer

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
        linedDesc.colorAttachments[0].isBlendingEnabled = true
        linedDesc.colorAttachments[0].rgbBlendOperation = .add
        linedDesc.colorAttachments[0].alphaBlendOperation = .add
        linedDesc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        linedDesc.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        linedDesc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        linedDesc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        linedDesc.vertexDescriptor = vertexDesc
        linedDesc.stencilAttachmentPixelFormat = .stencil8

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
        gridDesc.colorAttachments[0].isBlendingEnabled = true
        gridDesc.colorAttachments[0].rgbBlendOperation = .add
        gridDesc.colorAttachments[0].alphaBlendOperation = .add
        gridDesc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        gridDesc.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        gridDesc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        gridDesc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        gridDesc.vertexDescriptor = vertexDesc
        gridDesc.stencilAttachmentPixelFormat = .stencil8

        do {
            cardGridPipelineState = try device.makeRenderPipelineState(descriptor: gridDesc)
        } catch {
            fatalError("Failed to create grid card pipeline: \(error)")
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
        // "Always pass, and replace the stencil value with the reference (1)"
        let stencilWrite = MTLStencilDescriptor()
        stencilWrite.stencilCompareFunction = .always
        stencilWrite.stencilFailureOperation = .keep
        stencilWrite.depthFailureOperation = .keep
        stencilWrite.depthStencilPassOperation = .replace
        desc.frontFaceStencil = stencilWrite
        desc.backFaceStencil = stencilWrite
        stencilStateWrite = device.makeDepthStencilState(descriptor: desc)

        // 2. READ State (Used when drawing Card Strokes)
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

    // MARK: - Touch Handling

    func handleTouchBegan(at point: CGPoint, touchType: UITouch.TouchType) {
        //  MODAL INPUT: Only allow Pencil for drawing
        guard touchType == .pencil else { return }

        guard let view = metalView else { return }

        // USE NEW HIERARCHICAL HIT TEST
        // This checks children (foreground), active frame (middle), and parent (background)
        if let result = hitTestHierarchy(screenPoint: point, viewSize: view.bounds.size) {
            // Found a card (in active, parent, or child frame)
            // Store BOTH the card AND the frame it belongs to for correct coordinate transforms
            currentDrawingTarget = .card(result.card, result.frame)

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

        // Keep points in SCREEN space during drawing
        currentTouchPoints = [point]
    }

    func handleTouchMoved(at point: CGPoint, predicted: [CGPoint], touchType: UITouch.TouchType) {
        //  MODAL INPUT: Only allow Pencil for drawing
        guard touchType == .pencil else { return }

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
        //  MODAL INPUT: Only allow Pencil for drawing
        guard touchType == .pencil, let target = currentDrawingTarget else { return }

        guard let view = metalView else { return }

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
        }

        //  MODAL INPUT: Route stroke to correct target
        switch target {
        case .canvas(let frame):
            // DRAW ON CANVAS (existing logic)
            let stroke = Stroke(screenPoints: smoothScreenPoints,
                                zoomAtCreation: zoomScale,
                                panAtCreation: panOffset,
                                viewSize: view.bounds.size,
                                rotationAngle: rotationAngle,
                                color: brushSettings.color,
                                baseWidth: brushSettings.size,
                                device: device)
            frame.strokes.append(stroke)
            frame.markTilesDirty(for: stroke)

        case .card(let card, let frame):
            // DRAW ON CARD (Cross-Depth Compatible)
            // Transform points into card-local space accounting for which frame the card is in
            let cardStroke = createStrokeForCard(
                screenPoints: smoothScreenPoints,
                card: card,
                frame: frame,
                viewSize: view.bounds.size
            )
            card.strokes.append(cardStroke)
            frame.markTilesDirty(for: cardStroke)
        }

        currentTouchPoints = []
        liveStrokeOrigin = nil
        currentDrawingTarget = nil
        lastSavedPoint = nil  // Clear for next stroke
    }

    func handleTouchCancelled(touchType: UITouch.TouchType) {
        //  MODAL INPUT: Only allow Pencil for drawing
        guard touchType == .pencil else { return }
        predictedTouchPoints = []  // Clear predictions
        currentTouchPoints = []
        liveStrokeOrigin = nil  // Clear temporary origin
        lastSavedPoint = nil  // Clear for next stroke
    }

    // MARK: - Card Management

    /// Hit test across the entire visible hierarchy (Parent -> Active -> Children)
    /// Returns: The Card, The Frame it belongs to, and the Coordinate Conversion Scale
    /// The conversion scale is used to translate movement deltas between coordinate systems:
    ///   - Parent cards: scale < 1.0 (move slower - parent coords are smaller)
    ///   - Active cards: scale = 1.0 (normal movement)
    ///   - Child cards: scale > 1.0 (move faster - child coords are larger)
    func hitTestHierarchy(screenPoint: CGPoint, viewSize: CGSize) -> (card: Card, frame: Frame, conversionScale: Double)? {

        // 1. Calculate Point in Active Frame (World Space)
        let pointActive = screenToWorldPixels_PureDouble(
            screenPoint,
            viewSize: viewSize,
            panOffset: panOffset,
            zoomScale: zoomScale,
            rotationAngle: rotationAngle
        )

        // --- CHECK 1: CHILDREN (Foreground - Top Priority) ---
        // Iterate children (reverse to hit top-most first)
        for child in activeFrame.children.reversed() {
            // Convert Active Point -> Child Point
            // Math: Translate to child origin, then Scale UP (child coords are huge)
            let pointInChild = (pointActive - child.originInParent) * child.scaleRelativeToParent

            for card in child.cards.reversed() {
                if card.hitTest(pointInFrame: pointInChild) {
                    // Conversion: 1 unit in Active = 'scale' units in Child
                    return (card, child, child.scaleRelativeToParent)
                }
            }
        }

        // --- CHECK 2: ACTIVE FRAME (Middle) ---
        for card in activeFrame.cards.reversed() {
            if card.hitTest(pointInFrame: pointActive) {
                return (card, activeFrame, 1.0)
            }
        }

        // --- CHECK 3: PARENT FRAME (Background) ---
        if let parent = activeFrame.parent {
            // Convert Active Point -> Parent Point
            // Math: Scale DOWN (parent coords are small), then Translate to Active's origin in Parent
            let pointInParent = activeFrame.originInParent + (pointActive / activeFrame.scaleRelativeToParent)

            for card in parent.cards.reversed() {
                if card.hitTest(pointInFrame: pointInParent) {
                    // Conversion: 1 unit in Active = (1/scale) units in Parent
                    return (card, parent, 1.0 / activeFrame.scaleRelativeToParent)
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
        if let result = hitTestHierarchy(screenPoint: point, viewSize: view.bounds.size) {
            // Found a card! Notify SwiftUI
            onEditCard?(result.card)
        }
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

    /// Create a stroke in card-local coordinates
    /// Transforms screen-space points into the card's local coordinate system
    /// This ensures the stroke "sticks" to the card when it's moved or rotated
    ///
    /// **CROSS-DEPTH COMPATIBLE:**
    /// This method now supports drawing on cards in different frames (Parent/Active/Child).
    /// It determines the scale factor between the active frame and the target frame,
    /// then applies the appropriate coordinate transforms.
    ///
    /// - Parameters:
    ///   - screenPoints: Raw screen-space touch points
    ///   - card: The card to draw on
    ///   - frame: The frame the card belongs to (may be parent, active, or child)
    ///   - viewSize: Screen dimensions
    /// - Returns: A stroke with points relative to card center
    func createStrokeForCard(screenPoints: [CGPoint], card: Card, frame: Frame, viewSize: CGSize) -> Stroke {
        // 1. Get the Card's World Position & Rotation (in its Frame)
        let cardOrigin = card.origin
        let cardRot = Double(card.rotation)

        // Pre-calculate rotation trig (inverse rotation to convert to card-local)
        let c = cos(-cardRot)
        let s = sin(-cardRot)

        // 2. Determine Scale Factor (Active -> Card's Frame)
        // This tells us how to convert coordinates from active frame to the card's frame
        var frameScale: Double = 1.0
        if frame === activeFrame {
            // Card is in active frame - no scaling needed
            frameScale = 1.0
        } else if let p = activeFrame.parent, frame === p {
            // Card is in parent frame - parent coords are smaller, so scale down
            frameScale = 1.0 / activeFrame.scaleRelativeToParent
        } else if activeFrame.children.contains(where: { $0 === frame }) {
            // Card is in child frame - child coords are larger, so scale up
            frameScale = frame.scaleRelativeToParent
        }

        // 3. Calculate Effective Zoom for the Card's Frame
        // This is critical for cross-depth drawing
        var effectiveZoom: Double
        if frame === activeFrame {
            effectiveZoom = zoomScale
        } else if let p = activeFrame.parent, frame === p {
            // Parent frame is zoomed IN from our perspective
            effectiveZoom = zoomScale * activeFrame.scaleRelativeToParent
        } else if let child = activeFrame.children.first(where: { $0 === frame }) {
            // Child frame is zoomed OUT from our perspective
            effectiveZoom = zoomScale / child.scaleRelativeToParent
        } else {
            effectiveZoom = zoomScale
        }

        // 4. Transform Screen Points -> Card Local Points
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

            // B. Active World -> Card's Frame World (Apply scale transform)
            var targetWorldPt = worldPtActive
            if frame !== activeFrame {
                if frameScale > 1.0 {
                    // Child Frame: Translate and Scale UP
                    targetWorldPt = (worldPtActive - frame.originInParent) * frame.scaleRelativeToParent
                } else {
                    // Parent Frame: Scale DOWN and Translate
                    targetWorldPt = activeFrame.originInParent + (worldPtActive / activeFrame.scaleRelativeToParent)
                }
            }

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

        // 5. Create the Stroke with Effective Zoom
        return Stroke(
            screenPoints: cardLocalPoints,   // Virtual screen space (world units * effectiveZoom)
            zoomAtCreation: effectiveZoom,   // Use effective zoom for the card's frame!
            panAtCreation: .zero,            // We handled position manually
            viewSize: .zero,                 // We handled centering manually
            rotationAngle: 0,                // We handled rotation manually
            color: brushSettings.color,      // Use brush settings color
            baseWidth: brushSettings.size,   // Use brush settings size
            device: device                   // Pass device for buffer caching
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
