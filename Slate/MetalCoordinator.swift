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
    var samplerState: MTLSamplerState!                  // Sampler for card textures
    var vertexBuffer: MTLBuffer!

    //  Stencil States for Card Clipping
    var stencilStateDefault: MTLDepthStencilState! // Default passthrough (no testing)
    var stencilStateWrite: MTLDepthStencilState!   // Writes 1s to stencil (card background)
    var stencilStateRead: MTLDepthStencilState!    // Only draws where stencil == 1 (card strokes)
    var stencilStateClear: MTLDepthStencilState!   // Writes 0s to stencil (cleanup)

    // MARK: - Modal Input: Pencil vs. Finger

    /// Tracks what object we are currently drawing on with the pencil
    enum DrawingTarget {
        case canvas(Frame)
        case card(Card)
    }
    var currentDrawingTarget: DrawingTarget?

    var currentTouchPoints: [CGPoint] = []  // Stored in SCREEN space during drawing
    var liveStrokeOrigin: SIMD2<Double>?    // Temporary origin for live stroke (Double precision)

    //  COMMIT 1: Telescoping Reference Frames
    // Instead of a flat array, we use a linked list of Frames for infinite zoom
    let rootFrame = Frame()           // The "Base Reality" - top level that cannot be zoomed out of
    lazy var activeFrame: Frame = rootFrame  // The current "Local Universe" we are viewing/editing

    weak var metalView: MTKView?

    //  UPGRADED: Store camera state as Double for infinite precision
    var panOffset: SIMD2<Double> = .zero {
        didSet {
            // Cast to Float only when updating tile manager
            tileManager.currentPanOffset = SIMD2<Float>(Float(panOffset.x), Float(panOffset.y))
        }
    }
    var zoomScale: Double = 1.0 {
        didSet {
            // Cast to Float only when updating tile manager
            tileManager.currentZoomScale = Float(zoomScale)
        }
    }
    var rotationAngle: Float = 0.0

    // MARK: - Tiling System (Phase 1: Debug Only)
    let tileManager = TileManager()

    override init() {
        super.init()
        device = MTLCreateSystemDefaultDevice()!
        commandQueue = device.makeCommandQueue()!
        makePipeLine()
        makeVertexBuffer()

        // Initialize tile manager state (cast Double to Float)
        tileManager.currentZoomScale = Float(zoomScale)
        tileManager.currentPanOffset = SIMD2<Float>(Float(panOffset.x), Float(panOffset.y))
    }

    /// Update tile manager with current view size (call when view size changes)
    func updateTileManagerViewSize(_ size: CGSize) {
        tileManager.viewSize = size
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
        encoder.setRenderPipelineState(pipelineState)
        for stroke in frame.strokes {
            guard !stroke.localVertices.isEmpty else { continue }

            let relativeOffsetDouble = stroke.origin - cameraCenterInThisFrame
            let relativeOffset = SIMD2<Float>(
                Float(relativeOffsetDouble.x),
                Float(relativeOffsetDouble.y)
            )

            var transform = StrokeTransform(
                relativeOffset: relativeOffset,
                zoomScale: Float(currentZoom),
                screenWidth: Float(viewSize.width),
                screenHeight: Float(viewSize.height),
                rotationAngle: currentRotation
            )

            drawStroke(stroke, with: &transform, encoder: encoder)
        }

        // 2.2: RENDER CARDS (Middle layer - on top of canvas strokes) 
        for card in frame.cards {
            // A. Calculate Position
            // Card lives in the Frame, so it moves with the Frame
            let relativeOffsetDouble = card.origin - cameraCenterInThisFrame
            let relativeOffset = SIMD2<Float>(Float(relativeOffsetDouble.x), Float(relativeOffsetDouble.y))

            // B. Handle Rotation
            // Cards have their own rotation property
            // Total Rotation = Camera Rotation + Card Rotation
            let finalRotation = currentRotation + card.rotation

            var transform = StrokeTransform(
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
                encoder.setVertexBytes(&transform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
                var c = color
                encoder.setFragmentBytes(&c, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)

            case .image(let texture):
                // Use textured pipeline (requires texture binding)
                encoder.setRenderPipelineState(cardPipelineState)
                encoder.setVertexBytes(&transform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
                encoder.setFragmentTexture(texture, index: 0)
                encoder.setFragmentSamplerState(samplerState, index: 0)

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

                for stroke in card.strokes {
                    guard !stroke.localVertices.isEmpty else { continue }
                    let strokeOffset = stroke.origin
                    var strokeTransform = StrokeTransform(
                        relativeOffset: offset + SIMD2<Float>(Float(strokeOffset.x), Float(strokeOffset.y)),
                        zoomScale: Float(currentZoom),
                        screenWidth: Float(viewSize.width),
                        screenHeight: Float(viewSize.height),
                        rotationAngle: totalRotation
                    )
                    drawStroke(stroke, with: &strokeTransform, encoder: encoder)
                }
            }

            //  STEP 3: CLEANUP STENCIL (Reset to 0 for next card)
            // Draw the card quad again with stencil clear mode
            encoder.setRenderPipelineState(cardSolidPipelineState)
            encoder.setDepthStencilState(stencilStateClear)
            encoder.setStencilReferenceValue(0)
            encoder.setVertexBytes(&transform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
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

            // 4. Render each stroke individually
            for stroke in child.strokes {
                guard !stroke.localVertices.isEmpty else { continue }

                //  FIX: Add the Stroke's own origin to the Frame's offset
                // Before, this was missing 'stroke.origin', collapsing everything to (0,0)
                let totalRelativeOffset = stroke.origin + frameOffsetInChildUnits

                var childTransform = StrokeTransform(
                    relativeOffset: SIMD2<Float>(Float(totalRelativeOffset.x), Float(totalRelativeOffset.y)),
                    zoomScale: Float(childZoom),
                    screenWidth: Float(viewSize.width),
                    screenHeight: Float(viewSize.height),
                    rotationAngle: currentRotation
                )

                drawStroke(stroke, with: &childTransform, encoder: encoder)
            }

            // Note: We do NOT recurse into grandchildren to avoid rendering depth ±2, ±3, etc.
            // Just immediate children (depth +1) is enough for visual continuity
        }
    }

    /// Helper to draw a stroke with a given transform.
    /// Reduces code duplication across parent/current/child rendering.
    func drawStroke(_ stroke: Stroke, with transform: inout StrokeTransform, encoder: MTLRenderCommandEncoder) {
        guard !stroke.localVertices.isEmpty else { return }

        let vertexBuffer = device.makeBuffer(
            bytes: stroke.localVertices,
            length: stroke.localVertices.count * MemoryLayout<SIMD2<Float>>.stride,
            options: .storageModeShared
        )

        let transformBuffer = device.makeBuffer(
            bytes: &transform,
            length: MemoryLayout<StrokeTransform>.stride,
            options: .storageModeShared
        )

        encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        encoder.setVertexBuffer(transformBuffer, offset: 0, index: 1)
        encoder.drawPrimitives(type: .triangle,
                             vertexStart: 0,
                             vertexCount: stroke.localVertices.count)
    }

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
        let handleVertices: [StrokeVertex] = [
            StrokeVertex(position: SIMD2<Float>(-halfS, -halfS), uv: .zero),
            StrokeVertex(position: SIMD2<Float>(-halfS,  halfS), uv: .zero),
            StrokeVertex(position: SIMD2<Float>( halfS, -halfS), uv: .zero),
            StrokeVertex(position: SIMD2<Float>(-halfS,  halfS), uv: .zero),
            StrokeVertex(position: SIMD2<Float>( halfS, -halfS), uv: .zero),
            StrokeVertex(position: SIMD2<Float>( halfS,  halfS), uv: .zero)
        ]

        // 3. Calculate Corner Positions
        // We need to place these squares at the card's corners
        // IMPORTANT: The handles must rotate WITH the card
        let corners = card.getLocalCorners() // TL, TR, BL, BR

        // Setup Blue Color for Handles
        var color = SIMD4<Float>(0.0, 0.47, 1.0, 1.0) // iOS Blue
        var mode: Int = 1 // Solid Color
        encoder.setFragmentBytes(&color, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
        encoder.setFragmentBytes(&mode, length: MemoryLayout<Int>.stride, index: 1)

        // Reuse a buffer for the geometry (since all 4 handles are same size)
        let vertexBuffer = device.makeBuffer(bytes: handleVertices,
                                             length: handleVertices.count * MemoryLayout<StrokeVertex>.stride,
                                             options: [])
        encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)

        // 4. Draw Loop - One handle at each corner
        for cornerLocal in corners {
            // Position Logic:
            // 1. Start at Card Origin (World Space)
            // 2. Add Corner Offset (Rotated by Card Rotation)

            // Calculate the Handle's World Origin on CPU
            let cardRot = card.rotation
            let c = cos(cardRot)
            let s = sin(cardRot)

            // Rotate the corner offset (e.g., -50, -50) by card rotation
            let rotatedCornerX = Double(cornerLocal.x) * Double(c) - Double(cornerLocal.y) * Double(s)
            let rotatedCornerY = Double(cornerLocal.x) * Double(s) + Double(cornerLocal.y) * Double(c)

            // Absolute World Position of this handle
            let handleOriginX = card.origin.x + rotatedCornerX
            let handleOriginY = card.origin.y + rotatedCornerY

            // Calculate Relative Offset for Shader
            let relX = handleOriginX - cameraCenter.x
            let relY = handleOriginY - cameraCenter.y

            var transform = StrokeTransform(
                relativeOffset: SIMD2<Float>(Float(relX), Float(relY)),
                zoomScale: Float(zoom),
                screenWidth: Float(viewSize.width),
                screenHeight: Float(viewSize.height),
                rotationAngle: rotation // Rotates the little square itself so it aligns with card
            )

            encoder.setVertexBytes(&transform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
        }
    }

    func draw(in view: MTKView) {
        // Update tile manager view size (only changes when view resizes)
        if tileManager.viewSize != view.bounds.size {
            tileManager.viewSize = view.bounds.size
        }

        // PHASE 2: Calculate Camera Center in World Space (Double precision)
        // This is the "View Center" - where the center of the screen is in the infinite world.
        let cameraCenterWorld = calculateCameraCenterWorld(viewSize: view.bounds.size)

        //  COMMIT 2: Start rendering pipeline
        let commandBuffer = commandQueue.makeCommandBuffer()!
        guard let rpd = view.currentRenderPassDescriptor else { return }
        let enc = commandBuffer.makeRenderCommandEncoder(descriptor: rpd)!
        enc.setRenderPipelineState(pipelineState)
        enc.setCullMode(.none)

        //  COMMIT 2: RECURSIVE RENDERING
        // Render all committed strokes using the recursive renderer
        // This will automatically render parent frames (background) before the active frame (foreground)
        renderFrame(activeFrame,
                   cameraCenterInThisFrame: cameraCenterWorld,
                   viewSize: view.bounds.size,
                   currentZoom: zoomScale,
                   currentRotation: rotationAngle,
                   encoder: enc,
                   excludedChild: nil) // Start with no exclusion

        //  COMMIT 2: LIVE STROKE RENDERING
        // Live strokes are rendered on top of all committed strokes (foreground)
        if currentTouchPoints.count >= 2, let tempOrigin = liveStrokeOrigin {
            let zoom = Double(zoomScale)
            let worldWidth = 10.0 / zoom

            var localPoints: [SIMD2<Float>]
            var liveTransform: StrokeTransform

            //  Handle card vs canvas drawing differently
            if case .card(let card) = currentDrawingTarget {
                // CARD DRAWING: Transform to card-local coordinates
                let firstScreenPt = currentTouchPoints[0]
                let cardOrigin = card.origin
                let cardRot = Double(card.rotation)
                let cameraRot = Double(rotationAngle)

                // Pre-calculate card inverse rotation
                let cc = cos(-cardRot)
                let ss = sin(-cardRot)

                // Pre-calculate camera inverse rotation
                let cCam = cos(cameraRot)
                let sCam = sin(cameraRot)

                localPoints = currentTouchPoints.map { pt in
                    // A. Screen → World (un-rotate by camera, apply zoom)
                    let dx = Double(pt.x) - Double(firstScreenPt.x)
                    let dy = Double(pt.y) - Double(firstScreenPt.y)

                    let unrotatedX = dx * cCam + dy * sCam
                    let unrotatedY = -dx * sCam + dy * cCam

                    let worldDx = unrotatedX / zoom
                    let worldDy = unrotatedY / zoom

                    // B. Add to first touch world position
                    let firstWorldPt = screenToWorldPixels_PureDouble(
                        firstScreenPt,
                        viewSize: view.bounds.size,
                        panOffset: panOffset,
                        zoomScale: zoomScale,
                        rotationAngle: rotationAngle
                    )
                    let worldX = firstWorldPt.x + worldDx
                    let worldY = firstWorldPt.y + worldDy

                    // C. World → Card Local (subtract card origin, un-rotate by card)
                    let cardDx = worldX - cardOrigin.x
                    let cardDy = worldY - cardOrigin.y

                    let localX = cardDx * cc - cardDy * ss
                    let localY = cardDx * ss + cardDy * cc

                    return SIMD2<Float>(Float(localX), Float(localY))
                }

                // Use "magic offset" approach for card strokes
                let distX = card.origin.x - cameraCenterWorld.x
                let distY = card.origin.y - cameraCenterWorld.y
                let c = cos(Double(-card.rotation))
                let s = sin(Double(-card.rotation))
                let magicOffsetX = distX * c - distY * s
                let magicOffsetY = distX * s + distY * c

                liveTransform = StrokeTransform(
                    relativeOffset: SIMD2<Float>(Float(magicOffsetX), Float(magicOffsetY)),
                    zoomScale: Float(zoomScale),
                    screenWidth: Float(view.bounds.width),
                    screenHeight: Float(view.bounds.height),
                    rotationAngle: rotationAngle + card.rotation // Total rotation
                )
            } else {
                // CANVAS DRAWING: Use original approach
                let firstScreenPt = currentTouchPoints[0]
                let angle = Double(rotationAngle)
                let c = cos(angle)
                let s = sin(angle)

                localPoints = currentTouchPoints.map { pt in
                    let dx = Double(pt.x) - Double(firstScreenPt.x)
                    let dy = Double(pt.y) - Double(firstScreenPt.y)

                    let unrotatedX = dx * c + dy * s
                    let unrotatedY = -dx * s + dy * c

                    let worldDx = unrotatedX / zoom
                    let worldDy = unrotatedY / zoom

                    return SIMD2<Float>(Float(worldDx), Float(worldDy))
                }

                let relativeOffsetDouble = tempOrigin - cameraCenterWorld
                liveTransform = StrokeTransform(
                    relativeOffset: SIMD2<Float>(Float(relativeOffsetDouble.x),
                                                 Float(relativeOffsetDouble.y)),
                    zoomScale: Float(zoomScale),
                    screenWidth: Float(view.bounds.width),
                    screenHeight: Float(view.bounds.height),
                    rotationAngle: rotationAngle
                )
            }

            // Tessellate in LOCAL space
            let localVertices = tessellateStrokeLocal(
                centerPoints: localPoints,
                width: Float(worldWidth)
            )

            guard !localVertices.isEmpty else {
                enc.endEncoding()
                commandBuffer.present(view.currentDrawable!)
                commandBuffer.commit()
                return
            }

            //  If drawing on a card, set up stencil clipping for the live preview
            if case .card(let card) = currentDrawingTarget {
                // STEP 1: Write card's stencil mask
                enc.setDepthStencilState(stencilStateWrite)
                enc.setStencilReferenceValue(1)
                enc.setRenderPipelineState(cardSolidPipelineState)

                // Calculate card transform
                let cardRelativeOffset = card.origin - cameraCenterWorld
                let cardRotation = rotationAngle + card.rotation
                var cardTransform = StrokeTransform(
                    relativeOffset: SIMD2<Float>(Float(cardRelativeOffset.x), Float(cardRelativeOffset.y)),
                    zoomScale: Float(zoomScale),
                    screenWidth: Float(view.bounds.width),
                    screenHeight: Float(view.bounds.height),
                    rotationAngle: cardRotation
                )

                // Draw card quad to stencil buffer
                enc.setVertexBytes(&cardTransform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
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
                // Drawing on canvas - no stencil clipping needed
                enc.setDepthStencilState(stencilStateDefault)
            }

            // Create buffers and render live stroke
            let vertexBuffer = device.makeBuffer(
                bytes: localVertices,
                length: localVertices.count * MemoryLayout<SIMD2<Float>>.stride,
                options: .storageModeShared
            )

            let transformBuffer = device.makeBuffer(
                bytes: &liveTransform,
                length: MemoryLayout<StrokeTransform>.stride,
                options: .storageModeShared
            )

            enc.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
            enc.setVertexBuffer(transformBuffer, offset: 0, index: 1)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: localVertices.count)

            // STEP 3: Clean up stencil if we used it
            if case .card(let card) = currentDrawingTarget {
                enc.setDepthStencilState(stencilStateClear)
                enc.setStencilReferenceValue(0)
                enc.setRenderPipelineState(cardSolidPipelineState)

                // Redraw card quad to clear stencil
                var cardTransform = StrokeTransform(
                    relativeOffset: SIMD2<Float>(Float((card.origin - cameraCenterWorld).x),
                                                 Float((card.origin - cameraCenterWorld).y)),
                    zoomScale: Float(zoomScale),
                    screenWidth: Float(view.bounds.width),
                    screenHeight: Float(view.bounds.height),
                    rotationAngle: rotationAngle + card.rotation
                )
                enc.setVertexBytes(&cardTransform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
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

        //  COMMIT 4: Update Debug HUD
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
            Camera: \(cameraPosText)
            """
        }
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func makePipeLine() {
        let library = device.makeDefaultLibrary()!

        // Stroke Pipeline
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction   = library.makeFunction(name: "vertex_main")
        desc.fragmentFunction = library.makeFunction(name: "fragment_main")
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
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

        // Calculate World Point (High Precision) for the Frame
        let worldPoint = screenToWorldPixels_PureDouble(
            point,
            viewSize: view.bounds.size,
            panOffset: panOffset,
            zoomScale: zoomScale,
            rotationAngle: rotationAngle
        )

        //  HIT TEST CARDS (Reverse order = Top first)
        var hitCard: Card? = nil
        for card in activeFrame.cards.reversed() {
            if card.hitTest(pointInFrame: worldPoint) {
                hitCard = card
                break
            }
        }

        if let card = hitCard {
            // DRAW ON CARD
            currentDrawingTarget = .card(card)
            // For card strokes, we'll transform points into card space later
            liveStrokeOrigin = worldPoint
        } else {
            // DRAW ON CANVAS
            currentDrawingTarget = .canvas(activeFrame)
            liveStrokeOrigin = worldPoint
        }

        // Keep points in SCREEN space during drawing
        currentTouchPoints = [point]

        // Debug tiling system (only when debug mode enabled)
        if tileManager.debugMode {
            let worldPointCG = CGPoint(x: worldPoint.x, y: worldPoint.y)
            let tileKey = tileManager.getTileKey(worldPoint: worldPointCG)
            let debugOutput = tileManager.debugInfo(worldPoint: worldPointCG,
                                                    screenPoint: point,
                                                    tileKey: tileKey)
            print("\n TOUCH BEGAN - Target: \(currentDrawingTarget != nil ? "Card" : "Canvas")")
            print(debugOutput)
        }
    }

    func handleTouchMoved(at point: CGPoint, touchType: UITouch.TouchType) {
        //  MODAL INPUT: Only allow Pencil for drawing
        guard touchType == .pencil else { return }

        // Keep points in SCREEN space during drawing (key for precision!)
        currentTouchPoints.append(point)

        // Debug tiling system (only when debug mode enabled, every 10th point)
        if tileManager.debugMode && currentTouchPoints.count % 10 == 0 {
            guard let view = metalView else { return }
            let worldPoint = screenToWorldPixels(point,
                                               viewSize: view.bounds.size,
                                               panOffset: SIMD2<Float>(Float(panOffset.x), Float(panOffset.y)),
                                               zoomScale: Float(zoomScale),
                                               rotationAngle: rotationAngle)
            let tileKey = tileManager.getTileKey(worldPoint: worldPoint)
            let debugOutput = tileManager.debugInfo(worldPoint: worldPoint,
                                                    screenPoint: point,
                                                    tileKey: tileKey)
            print("\n TOUCH MOVED (point \(currentTouchPoints.count))")
            print(debugOutput)
        }
    }

    func handleTouchEnded(at point: CGPoint, touchType: UITouch.TouchType) {
        //  MODAL INPUT: Only allow Pencil for drawing
        guard touchType == .pencil, let target = currentDrawingTarget else { return }

        guard let view = metalView else { return }

        // Keep final point in SCREEN space
        currentTouchPoints.append(point)

        // Debug tiling system (only when debug mode enabled)
        if tileManager.debugMode {
            let worldPoint = screenToWorldPixels(point,
                                               viewSize: view.bounds.size,
                                               panOffset: SIMD2<Float>(Float(panOffset.x), Float(panOffset.y)),
                                               zoomScale: Float(zoomScale),
                                               rotationAngle: rotationAngle)
            let tileKey = tileManager.getTileKey(worldPoint: worldPoint)
            let debugOutput = tileManager.debugInfo(worldPoint: worldPoint,
                                                    screenPoint: point,
                                                    tileKey: tileKey)
            print("\n TOUCH ENDED")
            print(debugOutput)
        }

        guard currentTouchPoints.count >= 4 else {
            currentTouchPoints = []
            liveStrokeOrigin = nil
            currentDrawingTarget = nil
            return
        }

        // Smooth the screen-space points
        let smoothScreenPoints = catmullRomPoints(points: currentTouchPoints,
                                                  closed: false,
                                                  alpha: 0.5,
                                                  segmentsPerCurve: 20)

        //  MODAL INPUT: Route stroke to correct target
        switch target {
        case .canvas(let frame):
            // DRAW ON CANVAS (existing logic)
            let stroke = Stroke(screenPoints: smoothScreenPoints,
                                zoomAtCreation: zoomScale,
                                panAtCreation: panOffset,
                                viewSize: view.bounds.size,
                                rotationAngle: rotationAngle,
                                color: SIMD4<Float>(1.0, 0.0, 0.0, 1.0))
            frame.strokes.append(stroke)

        case .card(let card):
            // DRAW ON CARD
            // Transform points into card-local space so strokes move with the card
            let cardStroke = createStrokeForCard(
                screenPoints: smoothScreenPoints,
                card: card,
                viewSize: view.bounds.size
            )
            card.strokes.append(cardStroke)
        }

        currentTouchPoints = []
        liveStrokeOrigin = nil
        currentDrawingTarget = nil
    }

    func handleTouchCancelled(touchType: UITouch.TouchType) {
        //  MODAL INPUT: Only allow Pencil for drawing
        guard touchType == .pencil else { return }
        currentTouchPoints = []
        liveStrokeOrigin = nil  // Clear temporary origin
    }

    // MARK: - Card Management

    /// Add a new card to the canvas at the camera center
    /// The card will be a solid color and can be selected/dragged/edited
    func addCard() {
        print(" addCard() called")
        guard let view = metalView else {
            print(" ERROR: metalView is nil!")
            return
        }
        print(" metalView exists")

        // 1. Calculate camera center in world coordinates (where the user is looking)
        let cameraCenterWorld = calculateCameraCenterWorld(viewSize: view.bounds.size)

        // 2. Create card at camera center with reasonable size
        // Size is in world units - at 1.0 zoom, this is ~300x200 pixels (increased for visibility)
        let cardSize = SIMD2<Double>(300.0, 200.0)

        // 3. Use neon pink for high visibility against cyan background
        let neonPink = SIMD4<Float>(1.0, 0.0, 1.0, 1.0)  // Bright magenta

        // 4. Create the card
        let card = Card(
            origin: cameraCenterWorld,
            size: cardSize,
            rotation: 0,
            type: .solidColor(neonPink)
        )

        // 5. Add to active frame
        activeFrame.cards.append(card)

        // 6. Debug output
        print(" ========== ADD CARD DEBUG ==========")
        print(" Card Position: (\(cameraCenterWorld.x), \(cameraCenterWorld.y))")
        print(" Card Size: \(cardSize.x) × \(cardSize.y) world units")
        print(" Card Color: Neon Pink (1.0, 0.0, 1.0, 1.0)")
        print(" Card Rotation: 0 radians")
        print(" Current Zoom: \(zoomScale)")
        print(" Current Pan: (\(panOffset.x), \(panOffset.y))")
        print(" Current Rotation: \(rotationAngle) radians")
        print(" Total Cards in Frame: \(activeFrame.cards.count)")
        print(" Card ID: \(card.id)")
        print(" ======================================")
    }

    /// Create a stroke in card-local coordinates
    /// Transforms screen-space points into the card's local coordinate system
    /// This ensures the stroke "sticks" to the card when it's moved or rotated
    ///
    /// - Parameters:
    ///   - screenPoints: Raw screen-space touch points
    ///   - card: The card to draw on
    ///   - viewSize: Screen dimensions
    /// - Returns: A stroke with points relative to card center
    func createStrokeForCard(screenPoints: [CGPoint], card: Card, viewSize: CGSize) -> Stroke {
        // 1. Get the Card's World Position & Rotation (in the current Frame)
        let cardOrigin = card.origin
        let cardRot = Double(card.rotation)

        // Pre-calculate rotation trig
        let c = cos(-cardRot) // Inverse rotation to un-apply card angle
        let s = sin(-cardRot)

        // 2. Transform Screen Points -> Card Local Points
        // We effectively treat the Card as a temporary "Universe" for these points
        var cardLocalPoints: [CGPoint] = []
        let currentZoom = Double(zoomScale) // Capture current zoom

        for screenPt in screenPoints {
            // A. Screen -> Frame World (Standard conversion)
            // This divides by zoom to get world units
            let worldPt = screenToWorldPixels_PureDouble(
                screenPt,
                viewSize: viewSize,
                panOffset: panOffset,
                zoomScale: zoomScale,
                rotationAngle: rotationAngle
            )

            // B. Frame World -> Card Local (World Units)
            // Translate (Subtract Card Origin)
            let dx = worldPt.x - cardOrigin.x
            let dy = worldPt.y - cardOrigin.y

            // Rotate (Un-rotate by Card Rotation)
            // x' = x*c - y*s
            // y' = x*s + y*c
            let localX = dx * c - dy * s
            let localY = dx * s + dy * c

            //  CRITICAL FIX: Scale up to "virtual screen space"
            // We multiply by zoom so when Stroke.init divides by zoom,
            // we get back to world units (localX, localY).
            // This allows Stroke.init to correctly calculate stroke width (10.0 / zoom).
            let virtualScreenX = localX * currentZoom
            let virtualScreenY = localY * currentZoom

            cardLocalPoints.append(CGPoint(x: virtualScreenX, y: virtualScreenY))
        }

        // 3. Create the Stroke
        //  The math:
        // - Geometry: (WorldPos * Zoom) / Zoom = WorldPos  Correct position
        // - Width: 10.0 / Zoom = Correct world width 
        return Stroke(
            screenPoints: cardLocalPoints,   // Virtual screen space (world units * zoom)
            zoomAtCreation: zoomScale,       // Actual zoom (will divide, canceling multiply)
            panAtCreation: .zero,            // We handled position manually
            viewSize: .zero,                 // We handled centering manually
            rotationAngle: 0,                // We handled rotation manually
            color: SIMD4<Float>(0.0, 0.0, 1.0, 1.0) // Blue for card strokes
        )
    }

    // MARK: - Phase 3: Tile-Local Current Stroke Rendering

    /// Render current stroke using tile-local tessellation to avoid precision issues.
    /// This is the critical function that enables gap-free drawing at any zoom level!
    func renderCurrentStrokeTileLocal(view: MTKView) -> [SIMD2<Float>] {
        guard currentTouchPoints.count >= 2 else { return [] }

        // 1. Calculate width in tile-local space
        let widthTile = tileManager.calculateTileLocalWidth(baseWidth: 10.0)

        // 2. Group touch points by tile AND track which tiles each point belongs to
        var pointsByTile: [TileKey: [CGPoint]] = [:]
        var pointToTile: [Int: TileKey] = [:]  // Index → TileKey mapping

        for (index, worldPoint) in currentTouchPoints.enumerated() {
            let tileKey = tileManager.getTileKey(worldPoint: worldPoint)
            let localPoint = tileManager.worldToTileLocal(worldPoint: worldPoint, tileKey: tileKey)
            pointsByTile[tileKey, default: []].append(localPoint)
            pointToTile[index] = tileKey
        }

        // DEBUG: Check for tile boundary crossings
        if tileManager.debugMode {
            var crossings = 0
            for i in 0..<(currentTouchPoints.count - 1) {
                if pointToTile[i] != pointToTile[i + 1] {
                    crossings += 1
                    print(" TILE BOUNDARY CROSSING at segment \(i)→\(i+1)")
                    print("   Point \(i): tile \(pointToTile[i]?.description ?? "?")")
                    print("   Point \(i+1): tile \(pointToTile[i+1]?.description ?? "?")")
                }
            }

            if pointsByTile.count > 1 {
                print(" Stroke spans \(pointsByTile.count) tiles, \(crossings) boundary crossings")
                print("   Points per tile: \(pointsByTile.mapValues { $0.count })")
            }
        }

        // 3. Tessellate each tile's segment in tile-local space
        var allLocalVertices: [(tileKey: TileKey, vertices: [SIMD2<Float>])] = []

        for (tileKey, localPoints) in pointsByTile {
            guard localPoints.count >= 2 else { continue }

            let vertices = tessellateStrokeLocal(
                centerPoints: localPoints,
                width: CGFloat(widthTile)
            )
            allLocalVertices.append((tileKey: tileKey, vertices: vertices))
        }

        // 4. Convert tile-local vertices to world space, then to NDC
        var finalVertices: [SIMD2<Float>] = []

        for (tileKey, vertices) in allLocalVertices {
            for vertex in vertices {
                // Convert tile-local vertex to world coordinates
                let localPoint = CGPoint(x: Double(vertex.x), y: Double(vertex.y))
                let worldPoint = tileManager.tileLocalToWorld(localPoint: localPoint, tileKey: tileKey)

                // Convert world to NDC (existing pipeline)
                let ndcVertex = worldPixelToNDC(
                    point: worldPoint,
                    viewSize: view.bounds.size,
                    panOffset: SIMD2<Float>(Float(panOffset.x), Float(panOffset.y)),
                    zoomScale: Float(zoomScale)
                )

                finalVertices.append(ndcVertex)
            }
        }

        // Debug output for Phase 3
        if tileManager.debugMode && currentTouchPoints.count >= 2 {
            print("\n PHASE 3: TILE-LOCAL STROKE RENDERING")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("Zoom: \(zoomScale)x (level \(tileManager.referenceLevel))")
            print("Tile Size (world): \(String(format: "%.9f", tileManager.tileSizeAtReferenceLevel)) px")
            print("Width (world): \(String(format: "%.9f", 10.0 / Double(zoomScale))) px")
            print("Width (tile-local): \(String(format: "%.6f", widthTile)) units")
            print("Touch Points: \(currentTouchPoints.count)")
            print("Tiles Spanned: \(pointsByTile.count)")
            print("Total Vertices: \(finalVertices.count)")

            // Detailed point analysis
            print("\nPoint Analysis:")
            for (index, worldPoint) in currentTouchPoints.prefix(3).enumerated() {
                let tileKey = pointToTile[index]!
                let localPoint = tileManager.worldToTileLocal(worldPoint: worldPoint, tileKey: tileKey)

                print("  Point \(index):")
                print("    World: (\(String(format: "%.3f", worldPoint.x)), \(String(format: "%.3f", worldPoint.y)))")
                print("    Tile: Level \(tileKey.level), Grid (\(tileKey.tx), \(tileKey.ty))")
                print("    Local [0,1024]: (\(String(format: "%.3f", localPoint.x)), \(String(format: "%.3f", localPoint.y)))")

                // Check if local coords are in valid range
                if localPoint.x < 0 || localPoint.x > 1024 || localPoint.y < 0 || localPoint.y > 1024 {
                    print("     OUT OF RANGE!")
                } else {
                    print("     Valid")
                }
            }

            // Tessellation check
            print("\nTessellation Results:")
            for (tileKey, localPoints) in pointsByTile.prefix(3) {
                print("  Tile Level \(tileKey.level), Grid (\(tileKey.tx), \(tileKey.ty)): \(localPoints.count) points")
                if localPoints.count < 2 {
                    print("     INSUFFICIENT POINTS (need 2+)")
                }
            }

            // Coordinate magnitude check
            if let firstTile = allLocalVertices.first, let firstVertex = firstTile.vertices.first {
                print("\nCoordinate Magnitude Check:")
                print("  Tile-local vertex: (\(String(format: "%.3f", firstVertex.x)), \(String(format: "%.3f", firstVertex.y)))")
                let magnitude = max(abs(firstVertex.x), abs(firstVertex.y))
                print("  Max magnitude: \(String(format: "%.1f", magnitude))")

                if magnitude < 2000 {
                    print("   SMALL NUMBERS - Float precision preserved!")
                } else {
                    print("   LARGE NUMBERS - Float precision at risk!")
                }
            }
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        }

        return finalVertices
    }

    // MARK: - Phase 3 Diagnostics

    /// Diagnose gaps issue - test simple 2-point stroke
    func diagnoseGapsIssue() {
        print("\n GAP DIAGNOSIS")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        guard let view = metalView else { return }

        // Simulate a simple 2-point stroke at current zoom
        let screenP1 = CGPoint(x: 100, y: 100)
        let screenP2 = CGPoint(x: 200, y: 100)  // Horizontal line

        let worldP1 = screenToWorldPixels(screenP1, viewSize: view.bounds.size,
                                         panOffset: SIMD2<Float>(Float(panOffset.x), Float(panOffset.y)),
                                         zoomScale: Float(zoomScale),
                                         rotationAngle: rotationAngle)
        let worldP2 = screenToWorldPixels(screenP2, viewSize: view.bounds.size,
                                         panOffset: SIMD2<Float>(Float(panOffset.x), Float(panOffset.y)),
                                         zoomScale: Float(zoomScale),
                                         rotationAngle: rotationAngle)

        print("Test Stroke: Horizontal line, 100px on screen")
        print("  Screen P1: \(screenP1)")
        print("  Screen P2: \(screenP2)")
        print("  World P1: (\(String(format: "%.3f", worldP1.x)), \(String(format: "%.3f", worldP1.y)))")
        print("  World P2: (\(String(format: "%.3f", worldP2.x)), \(String(format: "%.3f", worldP2.y)))")
        print("  World distance: \(String(format: "%.3f", hypot(worldP2.x - worldP1.x, worldP2.y - worldP1.y))) px")

        // Check tile assignment
        let tile1 = tileManager.getTileKey(worldPoint: worldP1)
        let tile2 = tileManager.getTileKey(worldPoint: worldP2)
        let sameTile = (tile1 == tile2)

        print("\nTile Assignment:")
        print("  P1 tile: \(tile1.description)")
        print("  P2 tile: \(tile2.description)")
        print("  Same tile: \(sameTile ? " YES" : " NO - CROSSES BOUNDARY!")")

        // Test tile-local conversion
        let local1 = tileManager.worldToTileLocal(worldPoint: worldP1, tileKey: tile1)
        let local2 = tileManager.worldToTileLocal(worldPoint: worldP2, tileKey: tile2)

        print("\nTile-Local Coordinates:")
        print("  P1 local: (\(String(format: "%.3f", local1.x)), \(String(format: "%.3f", local1.y)))")
        print("  P2 local: (\(String(format: "%.3f", local2.x)), \(String(format: "%.3f", local2.y)))")

        if sameTile {
            let localDist = hypot(local2.x - local1.x, local2.y - local1.y)
            print("  Local distance: \(String(format: "%.3f", localDist))")

            // Test tessellation
            let widthTile = tileManager.calculateTileLocalWidth(baseWidth: 10.0)
            print("\nTessellation Test:")
            print("  Width (tile-local): \(String(format: "%.6f", widthTile))")

            let vertices = tessellateStrokeLocal(
                centerPoints: [local1, local2],
                width: CGFloat(widthTile)
            )
            print("  Generated vertices: \(vertices.count)")

            if vertices.isEmpty {
                print("   NO VERTICES GENERATED!")
            }
        } else {
            print("   Points in different tiles - segment will be LOST!")
        }

        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    }

    // MARK: - Phase 2 Testing

    /// Test tile-local tessellation with known inputs
    func testTileLocalTessellation() {
        print("\n TESTING TILE-LOCAL TESSELLATION")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        // Test case: Simple stroke in tile-local space
        let localPoints = [
            CGPoint(x: 100, y: 100),
            CGPoint(x: 200, y: 200),
            CGPoint(x: 300, y: 200)
        ]

        // Calculate tile-local width at current zoom
        let tileLocalWidth = tileManager.calculateTileLocalWidth(baseWidth: 10.0)

        print("Width Calculation:")
        print("  Zoom Scale: \(tileManager.currentZoomScale)")
        print("  Reference Level: \(tileManager.referenceLevel)")
        print("  Tile Size (world): \(String(format: "%.6f", tileManager.tileSizeAtReferenceLevel)) px")
        print("  World Width: \(String(format: "%.6f", 10.0 / Double(tileManager.currentZoomScale))) px")
        print("  Tile-Local Width: \(String(format: "%.6f", tileLocalWidth)) units")

        // Tessellate
        let vertices = tessellateStrokeLocal(
            centerPoints: localPoints,
            width: CGFloat(tileLocalWidth)
        )

        print("\nTessellation Results:")
        print("  Input Points: \(localPoints.count)")
        print("  Output Vertices: \(vertices.count)")

        if !vertices.isEmpty {
            let minX = vertices.map { $0.x }.min() ?? 0
            let maxX = vertices.map { $0.x }.max() ?? 0
            let minY = vertices.map { $0.y }.min() ?? 0
            let maxY = vertices.map { $0.y }.max() ?? 0

            print("  Vertex Range:")
            print("    X: [\(String(format: "%.2f", minX)), \(String(format: "%.2f", maxX))]")
            print("    Y: [\(String(format: "%.2f", minY)), \(String(format: "%.2f", maxY))]")

            // Check width offset from centerline
            let halfWidth = Float(tileLocalWidth / 2.0)
            print("  Expected Half-Width: \(String(format: "%.6f", halfWidth))")

            // Sample first quad (should be offset by halfWidth from p1)
            if vertices.count >= 6 {
                let p1 = SIMD2<Float>(Float(localPoints[0].x), Float(localPoints[0].y))
                let T1 = vertices[0]  // Top vertex of first segment
                let offset = distance(T1, p1)
                print("  Actual Offset from Center: \(String(format: "%.6f", offset))")

                let isCorrect = abs(offset - halfWidth) < 0.001
                print("   Width Correct: \(isCorrect)")
            }
        }

        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    }
}

// MARK: - Gesture Delegate

extension TouchableMTKView: UIGestureRecognizerDelegate {
    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer,
                           shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer) -> Bool {
        true
    }
}
