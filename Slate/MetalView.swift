import SwiftUI
import Metal
import MetalKit

// MARK: - Geometry / Tessellation

struct StrokeMesh {
    let origin: SIMD2<Float>
    let localVertices: [SIMD2<Float>]
}

/// Tessellate a stroke into triangle vertices expressed relative to a local origin.
func tessellateStroke(centerPoints: [CGPoint],
                      width: CGFloat,
                      segmentsPerCap: Int = 30) -> StrokeMesh {
    guard width > 0, let first = centerPoints.first else {
        return StrokeMesh(origin: .zero, localVertices: [])
    }

    let originDouble = SIMD2<Double>(Double(first.x), Double(first.y))
    let originFloat = SIMD2<Float>(Float(first.x), Float(first.y))

    // Degenerate circle stroke (single tap)
    if centerPoints.count == 1 {
        let circle = createCircleOffsets(at: first,
                                         radius: width / 2.0,
                                         segments: segmentsPerCap,
                                         origin: originDouble)
        return StrokeMesh(origin: originFloat, localVertices: circle)
    }

    var vertices: [SIMD2<Float>] = []
    vertices.reserveCapacity(centerPoints.count * segmentsPerCap * 3)

    let halfWidth = Double(width) * 0.5

    // Start cap
    vertices.append(contentsOf: createCircleOffsets(at: centerPoints[0],
                                                    radius: width / 2.0,
                                                    segments: segmentsPerCap,
                                                    origin: originDouble))

    for index in 0..<(centerPoints.count - 1) {
        let current = centerPoints[index]
        let next = centerPoints[index + 1]

        let worldP1 = SIMD2<Double>(Double(current.x), Double(current.y))
        let worldP2 = SIMD2<Double>(Double(next.x), Double(next.y))

        let dir = worldP2 - worldP1
        let lengthSq = dir.x * dir.x + dir.y * dir.y
        guard lengthSq > .ulpOfOne else { continue }

        let invLen = 1.0 / sqrt(lengthSq)
        let normal = dir * invLen
        let perp = SIMD2<Double>(-normal.y, normal.x)
        let offset = perp * halfWidth

        let t1 = worldP1 + offset - originDouble
        let b1 = worldP1 - offset - originDouble
        let t2 = worldP2 + offset - originDouble
        let b2 = worldP2 - offset - originDouble

        vertices.append(SIMD2<Float>(Float(t1.x), Float(t1.y)))
        vertices.append(SIMD2<Float>(Float(b1.x), Float(b1.y)))
        vertices.append(SIMD2<Float>(Float(t2.x), Float(t2.y)))

        vertices.append(SIMD2<Float>(Float(b1.x), Float(b1.y)))
        vertices.append(SIMD2<Float>(Float(b2.x), Float(b2.y)))
        vertices.append(SIMD2<Float>(Float(t2.x), Float(t2.y)))

        if index < centerPoints.count - 2 {
            vertices.append(contentsOf: createCircleOffsets(at: next,
                                                            radius: width / 2.0,
                                                            segments: 16,
                                                            origin: originDouble))
        }
    }

    if let lastPoint = centerPoints.last {
        vertices.append(contentsOf: createCircleOffsets(at: lastPoint,
                                                        radius: width / 2.0,
                                                        segments: segmentsPerCap,
                                                        origin: originDouble))
    }

    return StrokeMesh(origin: originFloat, localVertices: vertices)
}

/// Triangle fan circle expressed as offsets from the provided origin.
func createCircleOffsets(at point: CGPoint,
                         radius: CGFloat,
                         segments: Int = 30,
                         origin: SIMD2<Double>) -> [SIMD2<Float>] {
    guard radius > 0, segments >= 3 else { return [] }

    var vertices: [SIMD2<Float>] = []
    vertices.reserveCapacity(segments * 3)

    let centerWorld = SIMD2<Double>(Double(point.x), Double(point.y))
    let centerLocal = centerWorld - origin
    let radiusValue = Double(radius)
    let step = 2.0 * Double.pi / Double(segments)

    for i in 0..<segments {
        let angle0 = Double(i) * step
        let angle1 = Double(i + 1) * step

        let p1 = SIMD2<Double>(centerWorld.x + cos(angle0) * radiusValue,
                               centerWorld.y + sin(angle0) * radiusValue) - origin
        let p2 = SIMD2<Double>(centerWorld.x + cos(angle1) * radiusValue,
                               centerWorld.y + sin(angle1) * radiusValue) - origin

        vertices.append(SIMD2<Float>(Float(centerLocal.x), Float(centerLocal.y)))
        vertices.append(SIMD2<Float>(Float(p1.x), Float(p1.y)))
        vertices.append(SIMD2<Float>(Float(p2.x), Float(p2.y)))
    }

    return vertices
}

// MARK: - Coordinate Conversion Helpers

/// Screen pixels -> World pixels (inverse of worldToScreenPixels).
//func screenToWorldPixels(_ p: CGPoint,
//                         viewSize: CGSize,
//                         panOffset: SIMD2<Float>,
//                         zoomScale: Float,
//                         rotationAngle: Float) -> CGPoint {
//    let cx = Float(viewSize.width) * 0.5
//    let cy = Float(viewSize.height) * 0.5
//
//    let sx = Float(p.x), sy = Float(p.y)
//
//    let centeredX = sx - cx
//    let centeredY = sy - cy
//
//    let unpannedX = centeredX - panOffset.x
//    let unpannedY = centeredY - panOffset.y
//
//    let unzoomedX = unpannedX / zoomScale
//    let unzoomedY = unpannedY / zoomScale
//
//
////    x = x'×cos(-θ) + y'×sin(-θ) = x'×cos(θ) - y'×sin(θ)
////    y = -x'×sin(-θ) + y'×cos(θ) = x'×sin(θ) + y'×cos(θ)
//
//    let unrotatedX = unzoomedX * cos(rotationAngle) - unzoomedY * sin(rotationAngle)
//    let unrotatedY = unzoomedX * sin(rotationAngle) + unzoomedY * cos(rotationAngle)
//
//    let wx = unrotatedX + cx
//    let wy = unrotatedY + cy
//
//    return CGPoint(x: CGFloat(wx), y: CGFloat(wy))
//}

// Screen pixels -> World pixels (full inverse of the shader path in *NDC*)
func screenToWorldPixels(_ p: CGPoint,
                         viewSize: CGSize,
                         panOffset: SIMD2<Float>,
                         zoomScale: Float,
                         rotationAngle: Float) -> CGPoint {
    let W = Float(viewSize.width)
    let H = Float(viewSize.height)

    // screen -> NDC
    let ndcX = (Float(p.x) / W) * 2.0 - 1.0
    let ndcY = -((Float(p.y) / H) * 2.0 - 1.0)

    // unpan in NDC (NOTE: same conversion/signs as shader)
    let panNDCx = (panOffset.x / W) * 2.0
    let panNDCy = -(panOffset.y / H) * 2.0
    let upX = ndcX - panNDCx
    let upY = ndcY - panNDCy

    // unzoom
    let uzX = upX / zoomScale
    let uzY = upY / zoomScale

    // unrotate (R(-θ))
    let c = cos(rotationAngle), s = sin(rotationAngle)
    let posX =  uzX * c - uzY * s
    let posY =  uzX * s + uzY * c

    // NDC -> world pixels (same mapping you use for vertices)
    let wx = ((posX + 1.0) * 0.5) * W
    let wy = ((1.0 - posY) * 0.5) * H
    return CGPoint(x: CGFloat(wx), y: CGFloat(wy))
}

// World pixels -> Screen pixels
func worldToScreenPixels(_ w: CGPoint,
                         viewSize: CGSize,
                         panOffset: SIMD2<Float>,
                         zoomScale: Float,
                         rotationAngle: Float) -> CGPoint {
    let W = Float(viewSize.width)
    let H = Float(viewSize.height)

    // world pixels -> model NDC (identity)
    let x0 = (Float(w.x) / W) * 2.0 - 1.0
    let y0 = -((Float(w.y) / H) * 2.0 - 1.0)

    // rotate (R(θ)), same as shader
    let c = cos(rotationAngle), s = sin(rotationAngle)
    let rx =  x0 * c + y0 * s
    let ry = -x0 * s + y0 * c

    // zoom
    let zx = rx * zoomScale
    let zy = ry * zoomScale

    // pan in NDC (same conversion/signs as shader)
    let panNDCx = (panOffset.x / W) * 2.0
    let panNDCy = -(panOffset.y / H) * 2.0
    let ndcX = zx + panNDCx
    let ndcY = zy + panNDCy

    // NDC -> screen pixels
    let sx = ((ndcX + 1.0) * 0.5) * W
    let sy = ((1.0 - ndcY) * 0.5) * H
    return CGPoint(x: CGFloat(sx), y: CGFloat(sy))

}

@inline(__always)
func worldToModelNDC(_ w: CGPoint, viewSize: CGSize) -> SIMD2<Float> {
    let W = Float(viewSize.width), H = Float(viewSize.height)
    let x = (Float(w.x) / W) * 2.0 - 1.0
    let y = -((Float(w.y) / H) * 2.0 - 1.0)
    return SIMD2<Float>(x, y)
}

@inline(__always)
func screenToNDC(_ p: CGPoint, viewSize: CGSize) -> SIMD2<Float> {
    let W = Float(viewSize.width), H = Float(viewSize.height)
    let x = (Float(p.x) / W) * 2.0 - 1.0
    let y = -((Float(p.y) / H) * 2.0 - 1.0)
    return SIMD2<Float>(x, y)
}

/// Solve the pixel panOffset that keeps `anchorWorld` under `desiredScreen`
/// for the current zoom/rotation (matches shader math exactly).
func solvePanOffsetForAnchor(anchorWorld: SIMD2<Float>,
                             desiredScreen: CGPoint,
                             viewSize: CGSize,
                             zoomScale: Float,
                             rotationAngle: Float) -> SIMD2<Float> {
    let W = Float(viewSize.width), H = Float(viewSize.height)

    // modelNDC(w)
    let m = worldToModelNDC(CGPoint(x: CGFloat(anchorWorld.x), y: CGFloat(anchorWorld.y)),
                            viewSize: viewSize)

    // R(θ)*m
    let c = cos(rotationAngle), s = sin(rotationAngle)
    let rx =  m.x * c + m.y * s
    let ry = -m.x * s + m.y * c

    // desired screen → sNDC
    let sNDC = screenToNDC(desiredScreen, viewSize: viewSize)

    // panNDC = sNDC - zoom * R*m
    let panNDCx = sNDC.x - zoomScale * rx
    let panNDCy = sNDC.y - zoomScale * ry

    // Convert NDC pan to pixel panOffset (inverse of your shader’s conversion)
    // panNDC.x = (panPx / W)*2 → panPx = panNDC.x * (W/2)
    // panNDC.y = -(panPy / H)*2 → panPy = -panNDC.y * (H/2)
    let panPx = panNDCx * (W * 0.5)
    let panPy = -panNDCy * (H * 0.5)
    return SIMD2<Float>(panPx, panPy)
}



// MARK: - GPU Transform Struct

struct GPUTransform {
    var panOffset: SIMD2<Float>
    var zoomScale: Float
    var screenWidth: Float
    var screenHeight: Float
    var rotationAngle: Float
}

struct StrokeVertex {
    var localPosition: SIMD2<Float>
    var strokeIndex: UInt32
    var padding: UInt32 = 0
}

// MARK: - MetalView

struct MetalView: UIViewRepresentable {
    func makeUIView(context: Context) -> MTKView {
        let mtkView = TouchableMTKView()
        mtkView.device = MTLCreateSystemDefaultDevice()
        mtkView.clearColor = MTLClearColor(red: 0.1, green: 1.0, blue: 1.0, alpha: 1.0)
        mtkView.delegate = context.coordinator
        mtkView.isUserInteractionEnabled = true

        mtkView.coordinator = context.coordinator
        context.coordinator.metalView = mtkView

        return mtkView
    }

    func updateUIView(_ uiView: MTKView, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator() }

    // MARK: - TouchableMTKView
    class TouchableMTKView: MTKView {
        weak var coordinator: Coordinator?

        var panGesture: UIPanGestureRecognizer!
        var pinchGesture: UIPinchGestureRecognizer!
        var rotationGesture: UIRotationGestureRecognizer!

        // Pinch anchor (persist between .began and subsequent states)
        var pinchAnchorScreen: CGPoint = .zero
        var pinchAnchorWorld: SIMD2<Float> = .zero
        var panOffsetAtPinchStart: SIMD2<Float> = .zero
        
        //roation anchor
        var rotationAnchorScreen: CGPoint = .zero
        var rotationAnchorWorld: SIMD2<Float> = .zero
        var panOffsetAtRotationStart: SIMD2<Float> = .zero
        
        enum AnchorOwner { case none, pinch, rotation }

        var activeOwner: AnchorOwner = .none
        var anchorWorld: SIMD2<Float> = .zero
        var anchorScreen: CGPoint = .zero
        
        var lastPinchTouchCount: Int = 0
        var lastRotationTouchCount: Int = 0


        
        
        func lockAnchor(owner: AnchorOwner, at screenPt: CGPoint, coord: Coordinator) {
            activeOwner = owner
            anchorScreen = screenPt
            let w = screenToWorldPixels(screenPt,
                                        viewSize: bounds.size,
                                        panOffset: coord.panOffset,
                                        zoomScale: coord.zoomScale,
                                        rotationAngle: coord.rotationAngle)
            anchorWorld = SIMD2<Float>(Float(w.x), Float(w.y))
        }

        // Re-lock anchor to a new screen point *without changing the transform*
        func relockAnchorAtCurrentCentroid(owner: AnchorOwner, screenPt: CGPoint, coord: Coordinator) {
            activeOwner = owner
            anchorScreen = screenPt
            // IMPORTANT: recompute world under the *new* centroid using current pan/zoom/rotation.
            let w = screenToWorldPixels(screenPt,
                                        viewSize: bounds.size,
                                        panOffset: coord.panOffset,
                                        zoomScale: coord.zoomScale,
                                        rotationAngle: coord.rotationAngle)
            anchorWorld = SIMD2<Float>(Float(w.x), Float(w.y))
        }

        func handoffAnchor(to newOwner: AnchorOwner, screenPt: CGPoint, coord: Coordinator) {
            relockAnchorAtCurrentCentroid(owner: newOwner, screenPt: screenPt, coord: coord)
        }

        func clearAnchorIfUnused() { activeOwner = .none }


        

        override init(frame: CGRect, device: MTLDevice?) {
            super.init(frame: frame, device: device)
            setupGestures()
        }
        required init(coder: NSCoder) {
            super.init(coder: coder)
            setupGestures()
        }

        func setupGestures() {
            panGesture = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
            panGesture.minimumNumberOfTouches = 2
            panGesture.maximumNumberOfTouches = 2
            addGestureRecognizer(panGesture)

            pinchGesture = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
            addGestureRecognizer(pinchGesture)
            
            rotationGesture = UIRotationGestureRecognizer(target: self, action: #selector(handleRotation(_:)))
            addGestureRecognizer(rotationGesture)

            panGesture.delegate = self
            pinchGesture.delegate = self
            rotationGesture.delegate = self
        }

        @objc func handlePinch(_ gesture: UIPinchGestureRecognizer) {
            guard let coord = coordinator else { return }
            let loc = gesture.location(in: self)
            let tc  = gesture.numberOfTouches

            switch gesture.state {
            case .began:
                lastPinchTouchCount = tc
                if activeOwner == .none { lockAnchor(owner: .pinch, at: loc, coord: coord) }

            case .changed:
                // If touch count changed (finger lifted/added), re-lock to new centroid w/o moving content.
                if activeOwner == .pinch, tc != lastPinchTouchCount {
                    relockAnchorAtCurrentCentroid(owner: .pinch, screenPt: loc, coord: coord)
                    lastPinchTouchCount = tc
                    // Do not solve pan on this exact frame; early-return to avoid visible snap.
                    gesture.scale = 1.0
                    return
                }

                // Normal incremental zoom
                coord.zoomScale = coord.zoomScale * Float(gesture.scale)
                gesture.scale = 1.0

                // Keep the shared anchor pinned
                let target = (activeOwner == .pinch) ? loc : anchorScreen
                coord.panOffset = solvePanOffsetForAnchor(anchorWorld: anchorWorld,
                                                          desiredScreen: target,
                                                          viewSize: bounds.size,
                                                          zoomScale: coord.zoomScale,
                                                          rotationAngle: coord.rotationAngle)
                if activeOwner == .pinch { anchorScreen = target }

            case .ended, .cancelled, .failed:
                if activeOwner == .pinch {
                    // If rotation is active, hand off smoothly to its centroid
                    if rotationGesture.state == .changed || rotationGesture.state == .began {
                        let rloc = rotationGesture.location(in: self)
                        handoffAnchor(to: .rotation, screenPt: rloc, coord: coord)
                    } else {
                        clearAnchorIfUnused()
                    }
                }
                lastPinchTouchCount = 0

            default: break
            }
        }



        @objc func handlePan(_ gesture: UIPanGestureRecognizer) {
            let t = gesture.translation(in: self)
            coordinator?.panOffset.x += Float(t.x)
            coordinator?.panOffset.y += Float(t.y)
            gesture.setTranslation(.zero, in: self)

            if activeOwner != .none {
                anchorScreen.x += t.x
                anchorScreen.y += t.y
            }
        }

        
        @objc func handleRotation(_ gesture: UIRotationGestureRecognizer) {
            guard let coord = coordinator else { return }
            let loc = gesture.location(in: self)
            let tc  = gesture.numberOfTouches

            switch gesture.state {
            case .began:
                lastRotationTouchCount = tc
                if activeOwner == .none { lockAnchor(owner: .rotation, at: loc, coord: coord) }

            case .changed:
                // Re-lock when finger count changes to prevent jump.
                if activeOwner == .rotation, tc != lastRotationTouchCount {
                    relockAnchorAtCurrentCentroid(owner: .rotation, screenPt: loc, coord: coord)
                    lastRotationTouchCount = tc
                    gesture.rotation = 0.0
                    return
                }

                // Apply incremental rotation
                coord.rotationAngle += Float(gesture.rotation)
                gesture.rotation = 0.0

                // Keep shared anchor pinned
                let target = (activeOwner == .rotation) ? loc : anchorScreen
                coord.panOffset = solvePanOffsetForAnchor(anchorWorld: anchorWorld,
                                                          desiredScreen: target,
                                                          viewSize: bounds.size,
                                                          zoomScale: coord.zoomScale,
                                                          rotationAngle: coord.rotationAngle)
                if activeOwner == .rotation { anchorScreen = target }

            case .ended, .cancelled, .failed:
                if activeOwner == .rotation {
                    if pinchGesture.state == .changed || pinchGesture.state == .began {
                        let ploc = pinchGesture.location(in: self)
                        handoffAnchor(to: .pinch, screenPt: ploc, coord: coord)
                    } else {
                        clearAnchorIfUnused()
                    }
                }
                lastRotationTouchCount = 0

            default: break
            }
        }




        override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
            guard event?.allTouches?.count == 1, let touch = touches.first else { return }
            let location = touch.location(in: self)
            coordinator?.handleTouchBegan(at: location)
        }
        override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
            guard event?.allTouches?.count == 1, let touch = touches.first else { return }
            let location = touch.location(in: self)
            coordinator?.handleTouchMoved(at: location)
        }
        override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
            guard event?.allTouches?.count == 1, let touch = touches.first else { return }
            let location = touch.location(in: self)
            coordinator?.handleTouchEnded(at: location)
        }
        override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
            guard event?.allTouches?.count == 1 else { return }
            coordinator?.handleTouchCancelled()
        }
    }
}

// MARK: - Coordinator

class Coordinator: NSObject, MTKViewDelegate {
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    var pipelineState: MTLRenderPipelineState!
    var vertexBuffer: MTLBuffer!
    var originBuffer: MTLBuffer!

    var currentTouchPoints: [CGPoint] = []
    var allStrokes: [Stroke] = []

    weak var metalView: MTKView?

    var panOffset: SIMD2<Float> = .zero
    var zoomScale: Float = 1.0
    var rotationAngle: Float = 0.0

    override init() {
        super.init()
        device = MTLCreateSystemDefaultDevice()!
        commandQueue = device.makeCommandQueue()!
        makePipeLine()
        makeVertexBuffer()
        makeOriginBuffer()
    }

    func draw(in view: MTKView) {
        let startTime = Date()

        var vertices: [StrokeVertex] = []
        var origins: [SIMD2<Float>] = []

        for (strokeIndex, stroke) in allStrokes.enumerated() {
            let index = UInt32(strokeIndex)
            origins.append(stroke.origin)
            for local in stroke.localVertices {
                vertices.append(StrokeVertex(localPosition: local, strokeIndex: index))
            }
        }

        // Current stroke preview tessellated relative to its own origin
        if currentTouchPoints.count >= 2 {
            let mesh = tessellateStroke(
                centerPoints: currentTouchPoints,
                width: 10.0 / CGFloat(zoomScale),  // ← Fixed width in world pixels
            )
            if !mesh.localVertices.isEmpty {
                let previewIndex = UInt32(origins.count)
                origins.append(mesh.origin)
                for local in mesh.localVertices {
                    vertices.append(StrokeVertex(localPosition: local, strokeIndex: previewIndex))
                }
            }
        }

        let tessellationTime = Date().timeIntervalSince(startTime)
        if tessellationTime > 0.016 {
            print("⚠️ Tessellation taking \(tessellationTime * 1000)ms - too slow!")
        }

        // Transform buffer with current pan/zoom
        var transform = GPUTransform(
            panOffset: panOffset,
            zoomScale: zoomScale,
            screenWidth: Float(view.bounds.width),
            screenHeight: Float(view.bounds.height),
            rotationAngle: rotationAngle
        )

        updateVertexBuffer(with: vertices)
        updateOriginBuffer(with: origins)
        let commandBuffer = commandQueue.makeCommandBuffer()!
        guard let rpd = view.currentRenderPassDescriptor else { return }
        let enc = commandBuffer.makeRenderCommandEncoder(descriptor: rpd)!
        enc.setRenderPipelineState(pipelineState)
        enc.setCullMode(.none)

        if !vertices.isEmpty,
           let vertexBuffer = vertexBuffer,
           let originBuffer = originBuffer {
            enc.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
            enc.setVertexBuffer(originBuffer, offset: 0, index: 1)
            enc.setVertexBytes(&transform, length: MemoryLayout<GPUTransform>.stride, index: 2)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: vertices.count)
        }

        enc.endEncoding()
        commandBuffer.present(view.currentDrawable!)
        commandBuffer.commit()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func makePipeLine() {
        let library = device.makeDefaultLibrary()!
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction   = library.makeFunction(name: "vertex_main")
        desc.fragmentFunction = library.makeFunction(name: "fragment_main")
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        pipelineState = try? device.makeRenderPipelineState(descriptor: desc)
    }

    func makeVertexBuffer() {
        vertexBuffer = device.makeBuffer(length: MemoryLayout<StrokeVertex>.stride,
                                         options: [])
    }

    func makeOriginBuffer() {
        originBuffer = device.makeBuffer(length: MemoryLayout<SIMD2<Float>>.stride,
                                         options: [])
    }

    func updateVertexBuffer(with vertices: [StrokeVertex]) {
        guard !vertices.isEmpty else {
            vertexBuffer = nil
            return
        }
        let bufferSize = vertices.count * MemoryLayout<StrokeVertex>.stride
        vertexBuffer = device.makeBuffer(bytes: vertices, length: bufferSize, options: .storageModeShared)
    }

    func updateOriginBuffer(with origins: [SIMD2<Float>]) {
        guard !origins.isEmpty else {
            originBuffer = nil
            return
        }
        let bufferSize = origins.count * MemoryLayout<SIMD2<Float>>.stride
        originBuffer = device.makeBuffer(bytes: origins, length: bufferSize, options: .storageModeShared)
    }

    // MARK: - Touch Handling

    func handleTouchBegan(at point: CGPoint) {
        guard let view = metalView else { return }
        let worldPoint = screenToWorldPixels(point,
                                             viewSize: view.bounds.size,
                                             panOffset: panOffset,
                                             zoomScale: zoomScale, rotationAngle: rotationAngle)
        currentTouchPoints = [worldPoint]
    }

    func handleTouchMoved(at point: CGPoint) {
        guard let view = metalView else { return }
        let worldPoint = screenToWorldPixels(point,
                                             viewSize: view.bounds.size,
                                             panOffset: panOffset,
                                             zoomScale: zoomScale, rotationAngle: rotationAngle)
        currentTouchPoints.append(worldPoint)
    }

    func handleTouchEnded(at point: CGPoint) {
        guard let view = metalView else { return }
        let worldPoint = screenToWorldPixels(point,
                                             viewSize: view.bounds.size,
                                             panOffset: panOffset,
                                             zoomScale: zoomScale, rotationAngle: rotationAngle)
        currentTouchPoints.append(worldPoint)

        guard currentTouchPoints.count >= 4 else {
            currentTouchPoints = []
            return
        }

        let smoothPoints = catmullRomPoints(points: currentTouchPoints,
                                            closed: false,
                                            alpha: 0.5,
                                            segmentsPerCurve: 20)

        let stroke = Stroke(centerPoints: smoothPoints,
                            width: 10.0 / CGFloat(zoomScale),
                            color: SIMD4<Float>(1.0, 0.0, 0.0, 1.0),
                            viewSize: view.bounds.size)

        allStrokes.append(stroke)
        currentTouchPoints = []
    }

    func handleTouchCancelled() {
        currentTouchPoints = []
    }
}

// MARK: - Gesture Delegate

extension MetalView.TouchableMTKView: UIGestureRecognizerDelegate {
    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer,
                           shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer) -> Bool {
        true
    }
}
