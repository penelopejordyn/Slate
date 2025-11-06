import SwiftUI
import Metal
import MetalKit

func tessellateStroke(centerPoints: [CGPoint], width: CGFloat, viewSize: CGSize, panOffset: SIMD2<Float> = .zero, zoomScale: Float = 1.0) -> [SIMD2<Float>] {
    var vertices: [SIMD2<Float>] = []
    
    guard centerPoints.count >= 2 else {
        if centerPoints.count == 1 {
            return createCircle(at: centerPoints[0], radius: width / 2.0, viewSize: viewSize, panOffset: panOffset, zoomScale: zoomScale)
        }
        return vertices
    }
    
    let halfWidth = Float(width / 2.0)
    
    // 1. START CAP (round)
    let startCapVertices = createCircle(
        at: centerPoints[0],
        radius: width / 2.0,
        viewSize: viewSize,
        panOffset: panOffset,
        zoomScale: zoomScale
    )
    vertices.append(contentsOf: startCapVertices)
    
    // 2. MIDDLE SEGMENTS (flat quads) + JOINTS
    for i in 0..<(centerPoints.count - 1) {
        let current = centerPoints[i]
        let next = centerPoints[i + 1]
        
        // Convert to NDC (apply pan/zoom)
        let p1 = screenToNDC(point: current, viewSize: viewSize, panOffset: panOffset, zoomScale: zoomScale)
        let p2 = screenToNDC(point: next, viewSize: viewSize, panOffset: panOffset, zoomScale: zoomScale)
        
        // Calculate direction
        let direction = p2 - p1
        let length = sqrt(direction.x * direction.x + direction.y * direction.y)
        guard length > 0 else { continue }
        let normalized = direction / length
        
        // Get perpendicular
        let perpendicular = SIMD2<Float>(-normalized.y, normalized.x)
        
        // Scale width (in NDC)
        let widthInNDC = (halfWidth / Float(viewSize.width)) * 2.0 * zoomScale
        
        // Calculate edge points
        let T1 = p1 + perpendicular * widthInNDC
        let B1 = p1 - perpendicular * widthInNDC
        let T2 = p2 + perpendicular * widthInNDC
        let B2 = p2 - perpendicular * widthInNDC
        
        // Two triangles for this segment
        vertices.append(T1)
        vertices.append(B1)
        vertices.append(T2)
        
        vertices.append(B1)
        vertices.append(B2)
        vertices.append(T2)
        
        // 3. ADD JOINT at next point (if not the last point)
        if i < centerPoints.count - 2 {
            // Add a circle at the joint to fill gaps
            let jointVertices = createCircle(
                at: next,
                radius: width / 2.0,
                viewSize: viewSize,
                panOffset: panOffset,
                zoomScale: zoomScale,
                segments: 16  // Fewer segments for joints (optimization)
            )
            vertices.append(contentsOf: jointVertices)
        }
    }
    
    // 4. END CAP (round)
    let endCapVertices = createCircle(
        at: centerPoints[centerPoints.count - 1],
        radius: width / 2.0,
        viewSize: viewSize,
        panOffset: panOffset,
        zoomScale: zoomScale
    )
    vertices.append(contentsOf: endCapVertices)
    
    return vertices
}

// Helper function to create a circle
func createCircle(at point: CGPoint, radius: CGFloat, viewSize: CGSize, panOffset: SIMD2<Float> = .zero, zoomScale: Float = 1.0, segments: Int = 30) -> [SIMD2<Float>] {
    var vertices: [SIMD2<Float>] = []
    
    let center = screenToNDC(point: point, viewSize: viewSize, panOffset: panOffset, zoomScale: zoomScale)
    let radiusInNDC = (Float(radius) / Float(viewSize.width)) * 2.0 * zoomScale
    
    for i in 0..<segments {
        let angle1 = Float(i) * (2.0 * .pi / Float(segments))
        let angle2 = Float(i + 1) * (2.0 * .pi / Float(segments))
        
        let p1 = SIMD2<Float>(
            center.x + cos(angle1) * radiusInNDC,
            center.y + sin(angle1) * radiusInNDC
        )
        let p2 = SIMD2<Float>(
            center.x + cos(angle2) * radiusInNDC,
            center.y + sin(angle2) * radiusInNDC
        )
        
        vertices.append(center)
        vertices.append(p1)
        vertices.append(p2)
    }
    
    return vertices
}

func screenToNDC(point: CGPoint, viewSize: CGSize, panOffset: SIMD2<Float>, zoomScale: Float) -> SIMD2<Float> {
    // 1. Normalize to 0...1
    let normX = Float(point.x) / Float(viewSize.width)
    let normY = Float(point.y) / Float(viewSize.height)
    
    // 2. Center around 0.5
    let centeredX = normX - 0.5
    let centeredY = normY - 0.5
    
    // 3. Unapply zoom (divide because we're going backward)
    let worldX = centeredX / zoomScale
    let worldY = centeredY / zoomScale
    
    // 4. Unapply pan (subtract because we're going backward)
    //    Convert pan from pixels to normalized space first
    let panNormX = panOffset.x / Float(viewSize.width)
    let panNormY = panOffset.y / Float(viewSize.height)
    let pannedX = worldX - panNormX
    let pannedY = worldY - panNormY
    
    // 5. Convert to NDC (-1 to 1)
    let ndcX = pannedX * 2.0
    let ndcY = -pannedY * 2.0  // Flip Y
    
    return SIMD2<Float>(ndcX, ndcY)
}

func screenToWorld(point: CGPoint, viewSize: CGSize, panOffset: SIMD2<Float>, zoomScale: Float) -> CGPoint {
    // This must be the EXACT INVERSE of the GPU shader transform
    
    // 1. Convert screen pixels to NDC (-1 to 1)
    let ndcX = (Float(point.x) / Float(viewSize.width)) * 2.0 - 1.0
    let ndcY = -((Float(point.y) / Float(viewSize.height)) * 2.0 - 1.0)  // Flip Y
    
    // 2. Unapply pan (subtract what GPU adds)
    let panX = (panOffset.x / Float(viewSize.width)) * 2.0
    let panY = -(panOffset.y / Float(viewSize.height)) * 2.0
    let unpannedX = ndcX - panX
    let unpannedY = ndcY - panY
    
    // 3. Unapply zoom (divide what GPU multiplies)
    let worldX = unpannedX / zoomScale
    let worldY = unpannedY / zoomScale
    
    // 4. Convert back to screen pixels
    let screenX = ((worldX + 1.0) / 2.0) * Float(viewSize.width)
    let screenY = ((-(worldY) + 1.0) / 2.0) * Float(viewSize.height)  // Flip Y back
    
    return CGPoint(x: CGFloat(screenX), y: CGFloat(screenY))
}

struct MetalView: UIViewRepresentable {
    
    func makeUIView(context: Context) -> MTKView {
        let mtkView = TouchableMTKView()
        mtkView.device = MTLCreateSystemDefaultDevice()
        mtkView.clearColor = MTLClearColor(red: 0.1, green: 1.0, blue: 1.0, alpha: 1.0)
        mtkView.delegate = context.coordinator
        mtkView.isUserInteractionEnabled = true
        
        // Store coordinator reference so the view can communicate with it
        mtkView.coordinator = context.coordinator
        context.coordinator.metalView = mtkView
        
        return mtkView
    }
    
    func updateUIView(_ uiView: MTKView, context: Context) {

    }
    
    func makeCoordinator() -> Coordinator {
        return Coordinator()
    }
    
    // Custom MTKView subclass to handle touches
    class TouchableMTKView: MTKView {
        weak var coordinator: Coordinator?
        
        var panGesture: UIPanGestureRecognizer!
        var pinchGesture: UIPinchGestureRecognizer!
        
        // Then set it up in a function
        override init(frame: CGRect, device: MTLDevice?) {
            super.init(frame: frame, device: device)
            setupGestures()
        }
        
        required init(coder: NSCoder) {
            super.init(coder: coder)
            setupGestures()
        }
        
        func setupGestures() {
//             Pan gesture requires 2 fingers
            panGesture = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
            panGesture.minimumNumberOfTouches = 2
            panGesture.maximumNumberOfTouches = 2
            self.addGestureRecognizer(panGesture)
            
            // Pinch gesture automatically requires 2 fingers
            pinchGesture = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
            self.addGestureRecognizer(pinchGesture)
            
            // Allow both gestures to work simultaneously
            panGesture.delegate = self
            pinchGesture.delegate = self
        }
        
        @objc func handlePinch(_ gesture: UIPinchGestureRecognizer) {
            let scale = gesture.scale
            
            coordinator?.zoomScale *= Float(scale)
            
            gesture.scale = 1.0
        }
        @objc func handlePan(_ gesture: UIPanGestureRecognizer) {
            
            let translation = gesture.translation(in: self)
            
            let translationx = Float(translation.x)
            let translationy = Float(translation.y)
            
            // Corrected property name from `panOfffset` to `panOffset` on Coordinator
            coordinator?.panOffset.x += translationx
            coordinator?.panOffset.y += translationy
            
            // Consume the translation so next call provides the delta since we last handled it
            gesture.setTranslation(.zero, in: self)
            
        }
        
        override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
            // Only handle single touch for drawing
            guard event?.allTouches?.count == 1, let touch = touches.first else { return }
            let location = touch.location(in: self)
            coordinator?.handleTouchBegan(at: location)
        }
        
        override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
            // Only handle single touch for drawing
            guard event?.allTouches?.count == 1, let touch = touches.first else { return }
            let location = touch.location(in: self)
            coordinator?.handleTouchMoved(at: location)
        }
        
        override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
            // Only handle single touch for drawing
            guard event?.allTouches?.count == 1, let touch = touches.first else { return }
            let location = touch.location(in: self)
            coordinator?.handleTouchEnded(at: location)
        }
        
        override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
            guard event?.allTouches?.count == 1 else { return }
            coordinator?.handleTouchCancelled()
        }
        
    }
    
    // MARK: - Gesture Recognizer Delegate
    }
    
    // MARK: - Coordinator
    class Coordinator: NSObject, MTKViewDelegate {
        var device: MTLDevice!
        var commandQueue: MTLCommandQueue!
        var pipelineState: MTLRenderPipelineState!
        var vertexBuffer: MTLBuffer!
        var currentTouchPoints: [CGPoint] = []
        var allStrokes: [Stroke] = []
        weak var metalView: MTKView?
        var panOffset: SIMD2<Float> = .zero
        var zoomScale: Float = 1.0
        
        override init() {
            super.init()
            device = MTLCreateSystemDefaultDevice()!
            commandQueue = device.makeCommandQueue()!
            makePipeLine()
            makeVertexBuffer()
        }
        
        func draw(in view: MTKView) {
            let startTime = Date()
            
            var allVertices: [SIMD2<Float>] = []
            
            // Collect cached vertices
            for stroke in allStrokes {
                allVertices.append(contentsOf: stroke.vertices)
            }
            
            // Create transform buffer with screen size
            struct GPUTransform {
                var panOffset: SIMD2<Float>
                var zoomScale: Float
                var screenWidth: Float
                var screenHeight: Float
            }
            
            var transform = GPUTransform(
                panOffset: panOffset,
                zoomScale: zoomScale,
                screenWidth: Float(view.bounds.width),   // ← ADD
                screenHeight: Float(view.bounds.height)  // ← ADD
            )
            let transformBuffer = device.makeBuffer(
                bytes: &transform,
                length: MemoryLayout<GPUTransform>.stride,
                options: .storageModeShared
            )
            
            // Current stroke (no transform)
            if currentTouchPoints.count >= 2 {
                let currentVertices = tessellateStroke(
                    centerPoints: currentTouchPoints,
                    width: 10.0,
                    viewSize: view.bounds.size,
                    panOffset: .zero,
                    zoomScale: 1.0
                )
                allVertices.append(contentsOf: currentVertices)
            }
            
            let tessellationTime = Date().timeIntervalSince(startTime)
            if tessellationTime > 0.016 {
                print("⚠️ Tessellation taking \(tessellationTime * 1000)ms - too slow!")
            }

            
            // Draw test triangles OR strokes
            if allVertices.isEmpty {
                let commandBuffer = commandQueue.makeCommandBuffer()!
                guard let renderPassDescriptor = view.currentRenderPassDescriptor else { return }
                let renderPassEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
                renderPassEncoder.setRenderPipelineState(pipelineState)
                renderPassEncoder.setCullMode(.none)
                renderPassEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
                renderPassEncoder.setVertexBuffer(transformBuffer, offset: 0, index: 1)  // ← ADD THIS!
                renderPassEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
                renderPassEncoder.endEncoding()
                commandBuffer.present(view.currentDrawable!)
                commandBuffer.commit()
                return
            }
            
            // Draw strokes
            updateVertexBuffer(with: allVertices)
            
            let commandBuffer = commandQueue.makeCommandBuffer()!
            guard let renderPassDescriptor = view.currentRenderPassDescriptor else { return }
            let renderPassEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
            renderPassEncoder.setRenderPipelineState(pipelineState)
            renderPassEncoder.setCullMode(.none)
            renderPassEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
            renderPassEncoder.setVertexBuffer(transformBuffer, offset: 0, index: 1)
            renderPassEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: allVertices.count)
            renderPassEncoder.endEncoding()
            commandBuffer.present(view.currentDrawable!)
            commandBuffer.commit()
        }
        
            
        
        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            // Can leave empty for now
        }
        
        func makePipeLine() {
            let Library = device.makeDefaultLibrary()!
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.vertexFunction = Library.makeFunction(name: "vertex_main")
            pipelineDescriptor.fragmentFunction = Library.makeFunction(name: "fragment_main")
            pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            self.pipelineState = try? device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        }
        
        func makeVertexBuffer() {
            var positions: [SIMD2<Float>] = [
                // Triangle 1 (LEFT - magenta in shader)
                SIMD2<Float>(-0.8,  0.5),
                SIMD2<Float>(-0.3, -0.5),
                SIMD2<Float>(-0.8, -0.5),
                
                // Triangle 2 (RIGHT - should also be red)
                SIMD2<Float>(-0.3, -0.5),
                SIMD2<Float>(-0.3,  0.5),
                SIMD2<Float>(-0.8,  0.5),
            ]
            self.vertexBuffer = device.makeBuffer(bytes: &positions, length: positions.count * MemoryLayout<SIMD2<Float>>.stride, options: [])
        }
        
        // Touch handling methods
        // Touch handling methods
        // Touch handling methods
        // Touch handling methods
        func handleTouchBegan(at point: CGPoint) {
            guard let view = metalView else { return }
            
            // Convert screen touch to world coordinates
            let worldPoint = screenToWorld(
                point: point,
                viewSize: view.bounds.size,
                panOffset: panOffset,
                zoomScale: zoomScale
            )
            
            currentTouchPoints = []
            currentTouchPoints.append(worldPoint)  // ← Use worldPoint
            print("Touch began at screen: \(point), world: \(worldPoint)")
        }

        func handleTouchMoved(at point: CGPoint) {
            guard let view = metalView else { return }
            
            // Convert screen touch to world coordinates
            let worldPoint = screenToWorld(
                point: point,
                viewSize: view.bounds.size,
                panOffset: panOffset,
                zoomScale: zoomScale
            )
            
            print("Touch moved to screen: \(point), world: \(worldPoint)")
            currentTouchPoints.append(worldPoint)  // ← Use worldPoint
        }

        func handleTouchEnded(at point: CGPoint) {
            guard let view = metalView else { return }
            
            // Convert screen touch to world coordinates
            let worldPoint = screenToWorld(
                point: point,
                viewSize: view.bounds.size,
                panOffset: panOffset,
                zoomScale: zoomScale
            )
            
            print("Touch ended at screen: \(point), world: \(worldPoint)")
            currentTouchPoints.append(worldPoint)  // ← Use worldPoint
            
            guard currentTouchPoints.count >= 4 else {
                currentTouchPoints = []
                return
            }
            
            let smoothPoints = catmullRomPoints(
                points: currentTouchPoints,
                closed: false,
                alpha: 0.5,
                segmentsPerCurve: 20
            )
            
            let stroke = Stroke(
                centerPoints: smoothPoints,
                width: 10.0,
                color: SIMD4<Float>(1.0, 0.0, 0.0, 1.0),
                viewSize: view.bounds.size
            )
            
            allStrokes.append(stroke)
            currentTouchPoints = []
        }

        func handleTouchCancelled() {
            print("Touch cancelled")
            currentTouchPoints = []
        }
        

        func updateVertexBuffer(with vertices: [SIMD2<Float>]) {
            guard !vertices.isEmpty else { return }
            
            let bufferSize = vertices.count * MemoryLayout<SIMD2<Float>>.stride
            vertexBuffer = device.makeBuffer(
                bytes: vertices,
                length: bufferSize,
                options: .storageModeShared
            )
        }
    }


// MARK: - Gesture Recognizer Delegate
extension MetalView.TouchableMTKView: UIGestureRecognizerDelegate {
    // Allow pan and pinch to work together
    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer) -> Bool {
        return true
    }
}
