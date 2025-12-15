// MetalCoordinator.swift manages the Metal pipeline, render passes, camera state,
// and gesture-driven updates for the drawing experience.
import Foundation
import Metal
import MetalKit
import simd

private let DISAPPEAR_DELTA: Float = Float.greatestFiniteMagnitude
private let TILE_SIZE_PX: Int = 1024
private let TILE_PAD_PX: Int = 64
private let BASE_WORLD_UNITS_PER_PIXEL: Float = 1.0
private let MAX_TILE_CACHE = 300
private let MAX_VISIBLE_TILES = 900
private let MAX_BAKES_PER_FRAME = 5

// MARK: - Coordinator

fileprivate struct TileKey: Hashable {
    let frameID: UUID
    let level: Int
    let x: Int
    let y: Int
}

fileprivate struct TileEntry {
    var texture: MTLTexture
    var isDirty: Bool
    var hasContent: Bool
    var lastUsedFrame: Int
}

class Coordinator: NSObject, MTKViewDelegate {
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    var pipelineState: MTLRenderPipelineState!
    var cardPipelineState: MTLRenderPipelineState!      // Pipeline for textured cards
    var cardSolidPipelineState: MTLRenderPipelineState! // Pipeline for solid color cards
    var cardLinedPipelineState: MTLRenderPipelineState! // Pipeline for lined paper cards
    var cardGridPipelineState: MTLRenderPipelineState!  // Pipeline for grid paper cards
    var tilePipelineState: MTLRenderPipelineState!      // Pipeline for baked tiles
    var strokeSegmentPipelineState: MTLRenderPipelineState! // SDF segment pipeline
    var samplerState: MTLSamplerState!                  // Sampler for card textures
    var vertexBuffer: MTLBuffer!
    var quadVertexBuffer: MTLBuffer!                    // Unit quad for instanced segments
    fileprivate var tileCache: [TileKey: TileEntry] = [:]
    fileprivate var tileLRU: [TileKey] = []
    var frameCounter: Int = 0
    var tilesBakedThisFrame: Int = 0

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

    // MARK: - Brush Settings
    let brushSettings = BrushSettings()

    // MARK: - Debug Metrics
    var debugDrawnVerticesThisFrame: Int = 0
    var debugDrawnNodesThisFrame: Int = 0

    override init() {
        super.init()
        device = MTLCreateSystemDefaultDevice()!
        commandQueue = device.makeCommandQueue()!

        makePipeLine()
        makeVertexBuffer()
        makeQuadVertexBuffer()
    }

    // MARK: - Tile Helpers

    private func worldUnitsPerPixel(for level: Int) -> Double {
        return Double(BASE_WORLD_UNITS_PER_PIXEL) / pow(2.0, Double(level))
    }

    private func tileWorldSize(for level: Int) -> Double {
        return Double(TILE_SIZE_PX) * worldUnitsPerPixel(for: level)
    }

    private func tileLevel(for zoom: Double) -> Int {
        // Target world units per pixel ~ 1/zoom
        let target = 1.0 / max(zoom, 1e-12)
        let raw = log2(Double(BASE_WORLD_UNITS_PER_PIXEL) / target)
        let level = Int(round(raw))
        return max(-24, min(level, 12)) // allow larger tiles when zoomed out
    }

    private func tileKey(for frameID: UUID, level: Int, x: Int, y: Int) -> TileKey {
        TileKey(frameID: frameID, level: level, x: x, y: y)
    }

    private func touchTileLRU(_ key: TileKey) {
        if let idx = tileLRU.firstIndex(of: key) {
            tileLRU.remove(at: idx)
        }
        tileLRU.append(key)
        evictTilesIfNeeded()
    }

    private func evictTilesIfNeeded() {
        while tileCache.count > MAX_TILE_CACHE, let key = tileLRU.first {
            tileCache.removeValue(forKey: key)
            tileLRU.removeFirst()
        }
    }

    private func strokeWorldBounds(_ stroke: Stroke) -> CGRect {
        var rect = stroke.localBounds
        rect.origin.x += stroke.origin.x
        rect.origin.y += stroke.origin.y
        return rect
    }

    // Mark any cached tile that intersects this stroke as dirty, across all levels.
    private func markTilesDirty(for stroke: Stroke, in frame: Frame) {
        let strokeBounds = strokeWorldBounds(stroke)

        for (key, entry) in tileCache where key.frameID == frame.id {
            let tileRect = self.tileRect(for: key)
            if tileRect.intersects(strokeBounds) {
                var updated = entry
                updated.isDirty = true
                tileCache[key] = updated
            }
        }
    }

    private func visibleTiles(for frameID: UUID,
                              level: Int,
                              cameraCenter: SIMD2<Double>,
                              viewSize: CGSize,
                              zoom: Double) -> ([TileKey], Int) {
        let worldUnitsPerPixel = worldUnitsPerPixel(for: level)
        let halfWidth = Double(viewSize.width) * 0.5 / zoom
        let halfHeight = Double(viewSize.height) * 0.5 / zoom
        let padWorld = Double(TILE_PAD_PX) * worldUnitsPerPixel * 2.0
        // Pad for rotation and safety
        let pad = (halfWidth + halfHeight) * 0.5 + padWorld
        let minX = cameraCenter.x - halfWidth - pad
        let maxX = cameraCenter.x + halfWidth + pad
        let minY = cameraCenter.y - halfHeight - pad
        let maxY = cameraCenter.y + halfHeight + pad

        let tileSizeW = tileWorldSize(for: level)
        let tileMinX = Int(floor(minX / tileSizeW))
        let tileMaxX = Int(floor(maxX / tileSizeW))
        let tileMinY = Int(floor(minY / tileSizeW))
        let tileMaxY = Int(floor(maxY / tileSizeW))

        var keys: [TileKey] = []
        keys.reserveCapacity((tileMaxX - tileMinX + 1) * (tileMaxY - tileMinY + 1))
        for tx in tileMinX...tileMaxX {
            for ty in tileMinY...tileMaxY {
                keys.append(tileKey(for: frameID, level: level, x: tx, y: ty))
            }
        }

        // If too many tiles, pick a coarser level until under the cap
        if keys.count > MAX_VISIBLE_TILES {
            let newLevel = level - 1
            return visibleTiles(for: frameID, level: newLevel, cameraCenter: cameraCenter, viewSize: viewSize, zoom: zoom)
        }

        return (keys, level)
    }

    private func tileRect(for key: TileKey) -> CGRect {
        let size = tileWorldSize(for: key.level)
        let originX = Double(key.x) * size
        let originY = Double(key.y) * size
        return CGRect(x: originX, y: originY, width: size, height: size)
    }

    private func ensureTileTexture(for key: TileKey) -> MTLTexture? {
        if let existing = tileCache[key]?.texture {
            return existing
        }
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm,
                                                            width: TILE_SIZE_PX + TILE_PAD_PX * 2,
                                                            height: TILE_SIZE_PX + TILE_PAD_PX * 2,
                                                            mipmapped: false)
        desc.usage = [.renderTarget, .shaderRead]
        desc.storageMode = .private
        guard let tex = device.makeTexture(descriptor: desc) else { return nil }
        let entry = TileEntry(texture: tex, isDirty: true, hasContent: false, lastUsedFrame: frameCounter)
        tileCache[key] = entry
        touchTileLRU(key)
        return tex
    }

    private func bakeTileIfNeeded(key: TileKey, frame: Frame, level: Int, strokes: [Stroke], bakeCommandBuffer: MTLCommandBuffer) -> Bool {
        if tilesBakedThisFrame >= MAX_BAKES_PER_FRAME { return false }
        guard let texture = ensureTileTexture(for: key) else { return false }
        var entry = tileCache[key]!
        if !entry.isDirty { return false }
        tilesBakedThisFrame += 1

        let tileWorldRect = tileRect(for: key)
        let invWorldUnitsPerPixel = Float(1.0 / worldUnitsPerPixel(for: level))
        let currentZoomEffective = Float(invWorldUnitsPerPixel)
        let padWorld = Double(TILE_PAD_PX) * worldUnitsPerPixel(for: level)
        let expandedRect = tileWorldRect.insetBy(dx: -padWorld, dy: -padWorld)

        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = texture
        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].storeAction = .store
        rpd.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0)
        // Provide a stencil attachment because the stroke pipeline expects one
        let stencilDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .stencil8,
                                                                   width: TILE_SIZE_PX + TILE_PAD_PX * 2,
                                                                   height: TILE_SIZE_PX + TILE_PAD_PX * 2,
                                                                   mipmapped: false)
        stencilDesc.usage = .renderTarget
        stencilDesc.storageMode = .private
        if let stencilTex = device.makeTexture(descriptor: stencilDesc) {
            rpd.stencilAttachment.texture = stencilTex
            rpd.stencilAttachment.loadAction = .clear
            rpd.stencilAttachment.storeAction = .dontCare
        }

        guard let encoder = bakeCommandBuffer.makeRenderCommandEncoder(descriptor: rpd) else { return false }

        encoder.setRenderPipelineState(strokeSegmentPipelineState)
        encoder.setCullMode(.none)
        encoder.setDepthStencilState(stencilStateDefault)
        encoder.setVertexBuffer(quadVertexBuffer, offset: 0, index: 0)

        for stroke in strokes {
            guard let segmentBuffer = stroke.segmentBuffer, !stroke.segments.isEmpty else { continue }
            let delta = currentZoomEffective - stroke.zoomEffectiveAtCreation
            if delta >= DISAPPEAR_DELTA { continue }

            let strokeBounds = strokeWorldBounds(stroke)
            if !strokeBounds.intersects(expandedRect) { continue }

            let halfPixelWidth = max(Float(stroke.worldWidth) * currentZoomEffective * 0.5, 0.5)

            // Position stroke relative to tile center using high precision, then scale to tile pixels
            let tileCenterX = tileWorldRect.midX
            let tileCenterY = tileWorldRect.midY
            let dx = stroke.origin.x - tileCenterX
            let dy = stroke.origin.y - tileCenterY
            let angle: Double = 0 // baking in tile space, no rotation
            let c = cos(angle)
            let s = sin(angle)
            let rdx = dx * c - dy * s
            let rdy = dx * s + dy * c
            let rotatedOffsetScreen = SIMD2<Float>(Float(rdx * Double(invWorldUnitsPerPixel)),
                                                   Float(rdy * Double(invWorldUnitsPerPixel)))

            var transform = StrokeTransform(
                relativeOffset: .zero,
                rotatedOffsetScreen: rotatedOffsetScreen,
                zoomScale: invWorldUnitsPerPixel,
                screenWidth: Float(TILE_SIZE_PX + TILE_PAD_PX * 2),
                screenHeight: Float(TILE_SIZE_PX + TILE_PAD_PX * 2),
                rotationAngle: 0,
                halfPixelWidth: halfPixelWidth,
                featherPx: 1.0
            )

            encoder.setVertexBytes(&transform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
            encoder.setFragmentBytes(&transform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
            encoder.setVertexBuffer(segmentBuffer, offset: 0, index: 2)
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: stroke.segments.count)
        }

        encoder.endEncoding()

        entry.isDirty = false
        entry.hasContent = true
        entry.lastUsedFrame = frameCounter
        tileCache[key] = entry
        touchTileLRU(key)
        return true
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
                     bakeCommandBuffer: MTLCommandBuffer,
                     excludedChild: Frame? = nil) { //  NEW: Prevent double-rendering

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
                           bakeCommandBuffer: bakeCommandBuffer,
                           excludedChild: frame) //  TELL PARENT TO SKIP US
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

        // Convert cullRadius from screen pixels to world units for inverse culling

        let targetLevelRequested = tileLevel(for: currentZoom)
        let (targetKeys, targetLevelUsed) = visibleTiles(for: frame.id,
                                                         level: targetLevelRequested,
                                                         cameraCenter: cameraCenterInThisFrame,
                                                         viewSize: viewSize,
                                                         zoom: currentZoom)

        // Dual-layer rendering: draw a coarser parent level first as fallback, then draw the target level on top.
        // This prevents holes/flicker while higher-res tiles are being baked/throttled.
        let fallbackLevelRequested = targetLevelUsed - 1
        let (fallbackKeys, fallbackLevelUsed) = visibleTiles(for: frame.id,
                                                             level: fallbackLevelRequested,
                                                             cameraCenter: cameraCenterInThisFrame,
                                                             viewSize: viewSize,
                                                             zoom: currentZoom)

        func drawTileSet(keys: [TileKey], levelUsed: Int, allowBake: Bool) {
            guard !keys.isEmpty else { return }

            encoder.setRenderPipelineState(tilePipelineState)
            encoder.setDepthStencilState(stencilStateDefault)
            encoder.setFragmentSamplerState(samplerState, index: 0)

            let tileWorldSide = tileWorldSize(for: levelUsed)
            let tileHalfSide = tileWorldSide * 0.5

            // All tiles for a given level share the same world size and texture padding.
            let halfW = Float(tileHalfSide)
            let halfH = Float(tileHalfSide)
            let tileTexSide = Float(TILE_SIZE_PX + TILE_PAD_PX * 2)
            let padU = Float(TILE_PAD_PX) / tileTexSide
            let padV = Float(TILE_PAD_PX) / tileTexSide

            // NOTE: Tile textures are render targets; their sampling orientation differs from image cards.
            // Use v=0 at top (like the original tile path) to avoid vertical flipping.
            let tileVertices: [CardQuadVertex] = [
                CardQuadVertex(position: SIMD2<Float>(-halfW, -halfH), uv: SIMD2<Float>(padU, padV)),             // Top-Left
                CardQuadVertex(position: SIMD2<Float>(-halfW,  halfH), uv: SIMD2<Float>(padU, 1 - padV)),         // Bottom-Left
                CardQuadVertex(position: SIMD2<Float>( halfW, -halfH), uv: SIMD2<Float>(1 - padU, padV)),         // Top-Right
                CardQuadVertex(position: SIMD2<Float>(-halfW,  halfH), uv: SIMD2<Float>(padU, 1 - padV)),         // Bottom-Left
                CardQuadVertex(position: SIMD2<Float>( halfW,  halfH), uv: SIMD2<Float>(1 - padU, 1 - padV)),     // Bottom-Right
                CardQuadVertex(position: SIMD2<Float>( halfW, -halfH), uv: SIMD2<Float>(1 - padU, padV))          // Top-Right
            ]
            guard let tileVertexBuffer = device.makeBuffer(bytes: tileVertices,
                                                           length: tileVertices.count * MemoryLayout<CardQuadVertex>.stride,
                                                           options: .storageModeShared) else { return }
            encoder.setVertexBuffer(tileVertexBuffer, offset: 0, index: 0)

            let orderedKeys: [TileKey]
            if allowBake {
                // Prioritize tiles closest to the camera so throttling doesn't leave holes on-screen.
                orderedKeys = keys.sorted { a, b in
                    let ax = (Double(a.x) * tileWorldSide + tileHalfSide) - cameraCenterInThisFrame.x
                    let ay = (Double(a.y) * tileWorldSide + tileHalfSide) - cameraCenterInThisFrame.y
                    let bx = (Double(b.x) * tileWorldSide + tileHalfSide) - cameraCenterInThisFrame.x
                    let by = (Double(b.y) * tileWorldSide + tileHalfSide) - cameraCenterInThisFrame.y
                    return (ax * ax + ay * ay) < (bx * bx + by * by)
                }
            } else {
                orderedKeys = keys
            }

            for key in orderedKeys {
                if allowBake {
                    _ = bakeTileIfNeeded(key: key,
                                         frame: frame,
                                         level: levelUsed,
                                         strokes: frame.strokes,
                                         bakeCommandBuffer: bakeCommandBuffer)
                }

                guard var entry = tileCache[key] else { continue }
                // Never draw unbaked textures (they may contain undefined memory); rely on fallback instead.
                guard entry.hasContent else { continue }

                entry.lastUsedFrame = frameCounter
                tileCache[key] = entry
                touchTileLRU(key)

                let centerX = Double(key.x) * tileWorldSide + tileHalfSide
                let centerY = Double(key.y) * tileWorldSide + tileHalfSide
                let relativeOffset = SIMD2<Float>(Float(centerX - cameraCenterInThisFrame.x),
                                                  Float(centerY - cameraCenterInThisFrame.y))

                var transform = CardTransform(
                    relativeOffset: relativeOffset,
                    zoomScale: Float(currentZoom),
                    screenWidth: Float(viewSize.width),
                    screenHeight: Float(viewSize.height),
                    rotationAngle: currentRotation
                )

                encoder.setVertexBytes(&transform, length: MemoryLayout<CardTransform>.stride, index: 1)
                encoder.setFragmentTexture(entry.texture, index: 0)
                encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
            }
        }

        // Fallback layer (low-res) first.
        drawTileSet(keys: fallbackKeys, levelUsed: fallbackLevelUsed, allowBake: false)
        // Target layer (high-res) on top.
        drawTileSet(keys: targetKeys, levelUsed: targetLevelUsed, allowBake: true)

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
                encoder.setRenderPipelineState(strokeSegmentPipelineState)
                encoder.setDepthStencilState(stencilStateRead) // <--- READ MODE
                encoder.setStencilReferenceValue(1)
                encoder.setVertexBuffer(quadVertexBuffer, offset: 0, index: 0)

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
                    guard !stroke.segments.isEmpty, let segmentBuffer = stroke.segmentBuffer else { continue }
                    let strokeOffset = stroke.origin
                    let strokeRelativeOffset = offset + SIMD2<Float>(Float(strokeOffset.x), Float(strokeOffset.y))

                    // Calculate screen-space thickness for card strokes
                    let basePixelWidth = Float(stroke.worldWidth * currentZoom)
                    let halfPixelWidth = max(basePixelWidth * 0.5, 0.5)

                    var strokeTransform = StrokeTransform(
                        relativeOffset: strokeRelativeOffset,
                        rotatedOffsetScreen: SIMD2<Float>(Float((Double(strokeRelativeOffset.x) * cos(Double(totalRotation)) - Double(strokeRelativeOffset.y) * sin(Double(totalRotation))) * currentZoom),
                                                          Float((Double(strokeRelativeOffset.x) * sin(Double(totalRotation)) + Double(strokeRelativeOffset.y) * cos(Double(totalRotation))) * currentZoom)),
                        zoomScale: Float(currentZoom),
                        screenWidth: Float(viewSize.width),
                        screenHeight: Float(viewSize.height),
                        rotationAngle: totalRotation,
                        halfPixelWidth: halfPixelWidth,
                        featherPx: 1.0
                    )

                    encoder.setVertexBytes(&strokeTransform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
                    encoder.setFragmentBytes(&strokeTransform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
                    encoder.setVertexBuffer(segmentBuffer, offset: 0, index: 2)
                    encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: stroke.segments.count)
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
                        bakeCommandBuffer: bakeCommandBuffer,
                        excludedChild: frame)
        }
    }

    private func renderLiveStroke(view: MTKView, encoder enc: MTLRenderCommandEncoder, cameraCenterWorld: SIMD2<Double>, tempOrigin: SIMD2<Double>) {
        guard let target = currentDrawingTarget else { return }

        var screenPoints = currentTouchPoints
        if let last = screenPoints.last,
           let firstPredicted = predictedTouchPoints.first,
           last == firstPredicted {
            screenPoints.append(contentsOf: predictedTouchPoints.dropFirst())
        } else {
            screenPoints.append(contentsOf: predictedTouchPoints)
        }

        // Segment pipeline can draw a dot (p0 == p1), but skip when we have no points.
        guard let firstScreenPoint = screenPoints.first else { return }

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

        enc.setCullMode(.none)

        switch target {
        case .canvas:
            let zoom = max(zoomScale, 1e-6)
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

            let segments = Stroke.buildSegments(from: localPoints, color: brushSettings.color)
            guard !segments.isEmpty else { return }
            guard let segmentBuffer = device.makeBuffer(bytes: segments,
                                                        length: segments.count * MemoryLayout<StrokeSegmentInstance>.stride,
                                                        options: .storageModeShared) else { return }

            let dx = tempOrigin.x - cameraCenterWorld.x
            let dy = tempOrigin.y - cameraCenterWorld.y
            let rotatedOffsetScreen = SIMD2<Float>(
                Float((dx * c - dy * s) * zoom),
                Float((dx * s + dy * c) * zoom)
            )

            let halfPixelWidth = max(Float(brushSettings.size) * 0.5, 0.5)
            var transform = StrokeTransform(
                relativeOffset: .zero,
                rotatedOffsetScreen: rotatedOffsetScreen,
                zoomScale: Float(zoom),
                screenWidth: Float(view.bounds.size.width),
                screenHeight: Float(view.bounds.size.height),
                rotationAngle: rotationAngle,
                halfPixelWidth: halfPixelWidth,
                featherPx: 1.0
            )

            enc.setRenderPipelineState(strokeSegmentPipelineState)
            enc.setDepthStencilState(stencilStateDefault)
            enc.setVertexBuffer(quadVertexBuffer, offset: 0, index: 0)
            enc.setVertexBytes(&transform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
            enc.setFragmentBytes(&transform, length: MemoryLayout<StrokeTransform>.stride, index: 1)
            enc.setVertexBuffer(segmentBuffer, offset: 0, index: 2)
            enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: segments.count)

        case .card(let card, let frame):
            // Determine the camera transform in the target frame (parent/active/child).
            var cameraCenterInTarget = cameraCenterWorld
            var zoomInTarget = zoomScale
            if let parent = activeFrame.parent, frame === parent {
                cameraCenterInTarget = activeFrame.originInParent + (cameraCenterWorld / activeFrame.scaleRelativeToParent)
                zoomInTarget = zoomScale * activeFrame.scaleRelativeToParent
            } else if let child = activeFrame.children.first(where: { $0 === frame }) {
                cameraCenterInTarget = (cameraCenterWorld - child.originInParent) * child.scaleRelativeToParent
                zoomInTarget = zoomScale / child.scaleRelativeToParent
            }

            let liveStroke = createStrokeForCard(screenPoints: screenPoints,
                                                 card: card,
                                                 frame: frame,
                                                 viewSize: view.bounds.size)
            guard !liveStroke.segments.isEmpty, let segmentBuffer = liveStroke.segmentBuffer else { return }

            // 1) Write stencil for the card region without affecting color output.
            let relativeOffsetDouble = card.origin - cameraCenterInTarget
            let relativeOffset = SIMD2<Float>(Float(relativeOffsetDouble.x), Float(relativeOffsetDouble.y))
            let finalRotation = rotationAngle + card.rotation

            var cardTransform = CardTransform(
                relativeOffset: relativeOffset,
                zoomScale: Float(zoomInTarget),
                screenWidth: Float(view.bounds.size.width),
                screenHeight: Float(view.bounds.size.height),
                rotationAngle: finalRotation
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
            enc.setVertexBuffer(cardVertexBuffer, offset: 0, index: 0)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)

            // 2) Draw the live stroke clipped to stencil.
            enc.setRenderPipelineState(strokeSegmentPipelineState)
            enc.setDepthStencilState(stencilStateRead)
            enc.setStencilReferenceValue(1)
            enc.setVertexBuffer(quadVertexBuffer, offset: 0, index: 0)

            let totalRotation = rotationAngle + card.rotation
            let distX = card.origin.x - cameraCenterInTarget.x
            let distY = card.origin.y - cameraCenterInTarget.y
            let cCard = cos(Double(-card.rotation))
            let sCard = sin(Double(-card.rotation))
            let magicOffsetX = distX * cCard - distY * sCard
            let magicOffsetY = distX * sCard + distY * cCard
            let offset = SIMD2<Float>(Float(magicOffsetX), Float(magicOffsetY))

            let strokeOffset = liveStroke.origin
            let strokeRelativeOffset = offset + SIMD2<Float>(Float(strokeOffset.x), Float(strokeOffset.y))

            let basePixelWidth = Float(liveStroke.worldWidth * zoomInTarget)
            let halfPixelWidth = max(basePixelWidth * 0.5, 0.5)

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
                featherPx: 1.0
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
            enc.setVertexBuffer(cardVertexBuffer, offset: 0, index: 0)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)

            enc.setDepthStencilState(stencilStateDefault)
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
                rotatedOffsetScreen: SIMD2<Float>(Float((Double(relativeOffset.x) * cos(Double(rotation)) - Double(relativeOffset.y) * sin(Double(rotation))) * zoom),
                                                  Float((Double(relativeOffset.x) * sin(Double(rotation)) + Double(relativeOffset.y) * cos(Double(rotation))) * zoom)),
                zoomScale: Float(zoom),
                screenWidth: Float(viewSize.width),
                screenHeight: Float(viewSize.height),
                rotationAngle: rotation,
                halfPixelWidth: 1.0,  // Handles don't use thickness, but field is required
                featherPx: 1.0
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
        frameCounter += 1
        tilesBakedThisFrame = 0

        guard let bakeCommandBuffer = commandQueue.makeCommandBuffer() else { return }
        bakeCommandBuffer.label = "Tile Bake"

        guard let mainCommandBuffer = commandQueue.makeCommandBuffer(),
              let rpd = view.currentRenderPassDescriptor,
              let enc = mainCommandBuffer.makeRenderCommandEncoder(descriptor: rpd) else { return }

        let cameraCenterWorld = calculateCameraCenterWorld(viewSize: view.bounds.size)

        enc.setRenderPipelineState(strokeSegmentPipelineState)
        enc.setCullMode(.none)

        renderFrame(activeFrame,
                    cameraCenterInThisFrame: cameraCenterWorld,
                    viewSize: view.bounds.size,
                    currentZoom: zoomScale,
                    currentRotation: rotationAngle,
                    encoder: enc,
                    bakeCommandBuffer: bakeCommandBuffer,
                    excludedChild: nil)

        // Live stroke rendering (unchanged logic), reuse main encoder
        if (currentTouchPoints.count + predictedTouchPoints.count) >= 2, let tempOrigin = liveStrokeOrigin {
            renderLiveStroke(view: view, encoder: enc, cameraCenterWorld: cameraCenterWorld, tempOrigin: tempOrigin)
        }

        enc.endEncoding()
        bakeCommandBuffer.commit()
        if let drawable = view.currentDrawable {
            mainCommandBuffer.present(drawable)
        }
        mainCommandBuffer.commit()

        updateDebugHUD(view: view)
    }
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
            Vertices: \(self.debugDrawnVerticesThisFrame) | Draws: \(self.debugDrawnNodesThisFrame)
            """
        }
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

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
        let viewSampleCount = metalView?.sampleCount ?? 1

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
        segDesc.stencilAttachmentPixelFormat = .stencil8
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
        cardDesc.stencilAttachmentPixelFormat = .stencil8 //  Required for stencil buffer
        do {
            cardPipelineState = try device.makeRenderPipelineState(descriptor: cardDesc)
        } catch {
            fatalError("Failed to create cardPipelineState: \(error)")
        }

        // Tile Pipeline (dedicated vertex layout for tiles)
        let tileVertexDesc = MTLVertexDescriptor()
        tileVertexDesc.attributes[0].format = .float2
        tileVertexDesc.attributes[0].offset = 0
        tileVertexDesc.attributes[0].bufferIndex = 0
        tileVertexDesc.attributes[1].format = .float2
        tileVertexDesc.attributes[1].offset = MemoryLayout<SIMD2<Float>>.stride
        tileVertexDesc.attributes[1].bufferIndex = 0
        tileVertexDesc.layouts[0].stride = MemoryLayout<CardQuadVertex>.stride
        tileVertexDesc.layouts[0].stepFunction = .perVertex

        let tileDesc = MTLRenderPipelineDescriptor()
        tileDesc.vertexFunction   = library.makeFunction(name: "vertex_card")
        tileDesc.fragmentFunction = library.makeFunction(name: "fragment_card_texture")
        tileDesc.colorAttachments[0].pixelFormat = .bgra8Unorm
        tileDesc.sampleCount = viewSampleCount
        tileDesc.colorAttachments[0].isBlendingEnabled = true
        tileDesc.colorAttachments[0].rgbBlendOperation = .add
        tileDesc.colorAttachments[0].alphaBlendOperation = .add
        tileDesc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        tileDesc.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        tileDesc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        tileDesc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        tileDesc.vertexDescriptor = tileVertexDesc
        tileDesc.stencilAttachmentPixelFormat = .stencil8 // match main pass
        do {
            tilePipelineState = try device.makeRenderPipelineState(descriptor: tileDesc)
        } catch {
            fatalError("Failed to create tilePipelineState: \(error)")
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
        linedDesc.sampleCount = viewSampleCount
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
        gridDesc.sampleCount = viewSampleCount
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
	        // MODAL INPUT: Pencil on iPad; mouse/trackpad on Mac Catalyst.
	        guard isDrawingTouchType(touchType) else { return }

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
	        guard isDrawingTouchType(touchType), let target = currentDrawingTarget else { return }

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

            // Apply simplification to reduce vertex count
            smoothScreenPoints = simplifyStroke(
                smoothScreenPoints,
                minScreenDist: 1.5,
                minAngleDeg: 5.0
            )
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
                                zoomEffectiveAtCreation: Float(max(zoomScale, 1e-6)),
                                device: device)
            frame.strokes.append(stroke)
            markTilesDirty(for: stroke, in: frame)

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
        }

        currentTouchPoints = []
        liveStrokeOrigin = nil
        currentDrawingTarget = nil
        lastSavedPoint = nil  // Clear for next stroke
    }

	    func handleTouchCancelled(touchType: UITouch.TouchType) {
	        // MODAL INPUT: Pencil on iPad; mouse/trackpad on Mac Catalyst.
	        guard isDrawingTouchType(touchType) else { return }
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
            zoomAtCreation: max(effectiveZoom, 1e-6),   // Use effective zoom for the card's frame!
            panAtCreation: .zero,            // We handled position manually
            viewSize: .zero,                 // We handled centering manually
            rotationAngle: 0,                // We handled rotation manually
            color: brushSettings.color,      // Use brush settings color
            baseWidth: brushSettings.size,   // Use brush settings size
            zoomEffectiveAtCreation: Float(max(effectiveZoom, 1e-6)),
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
