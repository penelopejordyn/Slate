// TouchableMTKView.swift subclasses MTKView to attach gestures, manage anchors,
// and forward user interactions to the coordinator while presenting debug HUD info.
import UIKit
import MetalKit
import ObjectiveC.runtime
import simd

// MARK: - Associated Object Keys for gesture state storage
private struct AssociatedKeys {
    static var draggedCard: UInt8 = 0
}

// MARK: - TouchableMTKView
class TouchableMTKView: MTKView {
    weak var coordinator: Coordinator?

    var panGesture: UIPanGestureRecognizer!
    var pinchGesture: UIPinchGestureRecognizer!
    var rotationGesture: UIRotationGestureRecognizer!
    var longPressGesture: UILongPressGestureRecognizer!

    //  COMMIT 4: Debug HUD
    var debugLabel: UILabel!

    //  UPGRADED: Anchors now use Double for infinite precision at extreme zoom
    var pinchAnchorScreen: CGPoint = .zero
    var pinchAnchorWorld: SIMD2<Double> = .zero
    var panOffsetAtPinchStart: SIMD2<Double> = .zero

    //rotation anchor
    var rotationAnchorScreen: CGPoint = .zero
    var rotationAnchorWorld: SIMD2<Double> = .zero
    var panOffsetAtRotationStart: SIMD2<Double> = .zero

    enum AnchorOwner { case none, pinch, rotation }

    var activeOwner: AnchorOwner = .none
    var anchorWorld: SIMD2<Double> = .zero
    var anchorScreen: CGPoint = .zero

    var lastPinchTouchCount: Int = 0
    var lastRotationTouchCount: Int = 0




    func lockAnchor(owner: AnchorOwner, at screenPt: CGPoint, coord: Coordinator) {
        activeOwner = owner
        anchorScreen = screenPt

        //  FIX: Use Pure Double precision.
        // Previously this was using the Float version, causing the "Jump" on rotation.
        anchorWorld = screenToWorldPixels_PureDouble(screenPt,
                                                     viewSize: bounds.size,
                                                     panOffset: coord.panOffset, // SIMD2<Double>
                                                     zoomScale: coord.zoomScale, // Double
                                                     rotationAngle: coord.rotationAngle)
    }

    // Re-lock anchor to a new screen point *without changing the transform*
    func relockAnchorAtCurrentCentroid(owner: AnchorOwner, screenPt: CGPoint, coord: Coordinator) {
        activeOwner = owner
        anchorScreen = screenPt

        //  FIX: Use Pure Double precision here too.
        // This prevents jumps when you add/remove a finger (changing the centroid).
        anchorWorld = screenToWorldPixels_PureDouble(screenPt,
                                                     viewSize: bounds.size,
                                                     panOffset: coord.panOffset,
                                                     zoomScale: coord.zoomScale,
                                                     rotationAngle: coord.rotationAngle)
    }

    func handoffAnchor(to newOwner: AnchorOwner, screenPt: CGPoint, coord: Coordinator) {
        relockAnchorAtCurrentCentroid(owner: newOwner, screenPt: screenPt, coord: coord)
    }

    func clearAnchorIfUnused() { activeOwner = .none }

    // MARK: -  COMMIT 3: Telescoping Transitions

    /// Check if zoom has exceeded thresholds and perform frame transitions if needed.
    /// Returns TRUE if a transition occurred (caller should return early).
    func checkTelescopingTransitions(coord: Coordinator, currentCentroid: CGPoint) -> Bool {
        // DRILL DOWN: Zoom exceeded upper limit → Create child frame
        if coord.zoomScale > 1000.0 {
            //  Pass the shared anchor instead of recomputing
            drillDownToNewFrame(coord: coord,
                               anchorWorld: anchorWorld,
                               anchorScreen: anchorScreen)
            return true //  Transition happened
        }
        // POP UP: Zoom fell below lower limit → Return to parent frame
        else if coord.zoomScale < 0.5, coord.activeFrame.parent != nil {
            //  Pass the shared anchor instead of recomputing
            popUpToParentFrame(coord: coord,
                              anchorWorld: anchorWorld,
                              anchorScreen: anchorScreen)
            return true //  Transition happened
        }

        return false // No change
    }

    /// "The Silent Teleport" - Drill down into a child frame.
    ///  FIX: Uses the shared anchor instead of recomputing to prevent micro-jumps.
    func drillDownToNewFrame(coord: Coordinator,
                             anchorWorld: SIMD2<Double>,
                             anchorScreen: CGPoint) {
        // 1. CAPTURE STATE (CRITICAL: Do this BEFORE resetting zoom)
        let currentZoom = coord.zoomScale // This should be ~1000.0

        // 2. USE THE EXACT ANCHOR (Don't recompute from screen!)
        // This is the key fix - we use the exact same world point that the gesture handler
        // has been tracking, preventing floating point discrepancies.
        let pinchPointWorld = anchorWorld // EXACT same world point as gesture anchor
        let currentCentroid = anchorScreen // Reuse for solving pan

        // 3. Search radius: Keep it reasonable to avoid snapping to far-away frames
        let searchRadius: Double = 50.0

        var targetFrame: Frame? = nil
        for child in coord.activeFrame.children {
            if distance(child.originInParent, pinchPointWorld) < searchRadius {
                targetFrame = child
                break
            }
        }

        if let existing = targetFrame {
            //  RE-ENTER EXISTING FRAME
            coord.activeFrame = existing

            // 4. Calculate where the FINGER is inside this frame
            // LocalPinch = (ParentPinch - Origin) * Scale
            let diffX = pinchPointWorld.x - existing.originInParent.x
            let diffY = pinchPointWorld.y - existing.originInParent.y

            let localPinchX = diffX * existing.scaleRelativeToParent
            let localPinchY = diffY * existing.scaleRelativeToParent

            // 5. RESET ZOOM
            // We do this AFTER calculating positions
            coord.zoomScale = currentZoom / existing.scaleRelativeToParent

            // 6. SOLVE PAN
            coord.panOffset = solvePanOffsetForAnchor_Double(
                anchorWorld: SIMD2<Double>(localPinchX, localPinchY),
                desiredScreen: currentCentroid,
                viewSize: bounds.size,
                zoomScale: coord.zoomScale, // Now ~1.0
                rotationAngle: coord.rotationAngle
            )

            print(" Re-entered frame. Strokes: \(existing.strokes.count)")

        } else {
            //  CREATE NEW FRAME

            //  FIX: Center the new frame exactly on the PINCH POINT (Finger).
            // This prevents exponential coordinate growth (Off-Center Accumulation).
            // OLD: Centered on screen center → 500px offset compounds to 500,000 → 500M → 10^18 → CRASH
            // NEW: Centered on finger → offset resets to 0 at each depth → stays bounded forever
            let newFrameOrigin = pinchPointWorld

            let newFrame = Frame(
                parent: coord.activeFrame,
                origin: newFrameOrigin,
                scale: currentZoom // Use captured high zoom
            )
            coord.activeFrame.children.append(newFrame)

            coord.activeFrame = newFrame
            coord.zoomScale = 1.0

            //  RESULT: The pinch point is now the origin (0,0)
            // diffX = pinchPointWorld - newFrameOrigin = 0
            // diffY = pinchPointWorld - newFrameOrigin = 0
            // localPinch = (0, 0)

            coord.panOffset = solvePanOffsetForAnchor_Double(
                anchorWorld: SIMD2<Double>(0, 0), // Finger is at Local (0,0)
                desiredScreen: currentCentroid,   // Keep Finger at Screen Point
                viewSize: bounds.size,
                zoomScale: 1.0,
                rotationAngle: coord.rotationAngle
            )

            print(" Created NEW frame. Origin centered on pinch. Depth: \(frameDepth(coord.activeFrame))")
        }

        // 7. RE-ANCHOR GESTURES (update with new coordinate system)
        if activeOwner != .none {
            self.anchorWorld = screenToWorldPixels_PureDouble(
                currentCentroid,
                viewSize: bounds.size,
                panOffset: coord.panOffset,
                zoomScale: coord.zoomScale,
                rotationAngle: coord.rotationAngle
            )
            self.anchorScreen = currentCentroid
        }
    }

    /// Helper: Calculate Euclidean distance between two points
    func distance(_ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Double {
        let dx = b.x - a.x
        let dy = b.y - a.y
        return sqrt(dx * dx + dy * dy)
    }

    /// "The Reverse Teleport" - Pop up to the parent frame.
    ///  FIX: Uses the shared anchor instead of recomputing to prevent micro-jumps.
    func popUpToParentFrame(coord: Coordinator,
                            anchorWorld: SIMD2<Double>,
                            anchorScreen: CGPoint) {
        guard let parent = coord.activeFrame.parent else { return }

        let currentFrame = coord.activeFrame

        // 1. Calculate new zoom in parent space
        let newZoom = coord.zoomScale * currentFrame.scaleRelativeToParent

        // 2. USE THE EXACT ANCHOR (Don't recompute from screen!)
        // This is in the active (child) frame's coordinates
        let pinchPosInChild = anchorWorld
        let currentCentroid = anchorScreen

        // 3. Convert Child Pinch Position -> Parent Pinch Position
        // Parent = Origin + (Child / Scale)
        let pinchPosInParentX = currentFrame.originInParent.x + (pinchPosInChild.x / currentFrame.scaleRelativeToParent)
        let pinchPosInParentY = currentFrame.originInParent.y + (pinchPosInChild.y / currentFrame.scaleRelativeToParent)

        // 4. Solve Pan to lock FINGER position
        let newPanOffset = solvePanOffsetForAnchor_Double(
            anchorWorld: SIMD2<Double>(pinchPosInParentX, pinchPosInParentY),
            desiredScreen: currentCentroid,
            viewSize: bounds.size,
            zoomScale: newZoom,
            rotationAngle: coord.rotationAngle
        )

        // 5. THE HANDOFF - Switch to parent
        coord.activeFrame = parent
        coord.zoomScale = newZoom
        coord.panOffset = newPanOffset

        // 6. RE-ANCHOR GESTURES (update with new coordinate system)
        if activeOwner != .none {
            self.anchorWorld = screenToWorldPixels_PureDouble(
                currentCentroid,
                viewSize: bounds.size,
                panOffset: coord.panOffset,
                zoomScale: coord.zoomScale,
                rotationAngle: coord.rotationAngle
            )
            self.anchorScreen = currentCentroid
        }

        print(" Popped up to parent frame. Depth: \(frameDepth(coord.activeFrame))")
    }

    /// Helper: Calculate the depth of a frame (how many parents it has)
    func frameDepth(_ frame: Frame) -> Int {
        var depth = 0
        var current: Frame? = frame
        while current?.parent != nil {
            depth += 1
            current = current?.parent
        }
        return depth
    }




    override init(frame: CGRect, device: MTLDevice?) {
        super.init(frame: frame, device: device)
        setupGestures()
    }
    required init(coder: NSCoder) {
        super.init(coder: coder)
        setupGestures()
    }

    func setupGestures() {
        //  MODAL INPUT: PAN (Finger Only - 1 finger for card drag/canvas pan)
        panGesture = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
        panGesture.minimumNumberOfTouches = 1
        panGesture.maximumNumberOfTouches = 1
        // Crucial: Ignore Apple Pencil for panning/dragging
        panGesture.allowedTouchTypes = [NSNumber(value: UITouch.TouchType.direct.rawValue)]
        addGestureRecognizer(panGesture)

        //  MODAL INPUT: TAP (Finger Only - Select/Edit Cards)
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
        tapGesture.allowedTouchTypes = [NSNumber(value: UITouch.TouchType.direct.rawValue)]
        addGestureRecognizer(tapGesture)

        pinchGesture = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
        addGestureRecognizer(pinchGesture)

        rotationGesture = UIRotationGestureRecognizer(target: self, action: #selector(handleRotation(_:)))
        addGestureRecognizer(rotationGesture)

        longPressGesture = UILongPressGestureRecognizer(target: self, action: #selector(handleLongPress(_:)))
        longPressGesture.minimumPressDuration = 0.5
        longPressGesture.numberOfTouchesRequired = 2
        addGestureRecognizer(longPressGesture)

        panGesture.delegate = self
        pinchGesture.delegate = self
        rotationGesture.delegate = self
        longPressGesture.delegate = self

        //  COMMIT 4: Setup Debug HUD
        setupDebugHUD()
    }

    func setupDebugHUD() {
        debugLabel = UILabel()
        debugLabel.translatesAutoresizingMaskIntoConstraints = false
        debugLabel.font = UIFont.monospacedSystemFont(ofSize: 14, weight: .medium)
        debugLabel.textColor = .white
        debugLabel.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        debugLabel.numberOfLines = 0
        debugLabel.textAlignment = .left
        debugLabel.layer.cornerRadius = 8
        debugLabel.layer.masksToBounds = true
        debugLabel.text = "Frame: 0 | Zoom: 1.0×"
        debugLabel.isUserInteractionEnabled = false

        // Add padding to the label
        debugLabel.layoutMargins = UIEdgeInsets(top: 8, left: 12, bottom: 8, right: 12)

        addSubview(debugLabel)

        // Position in top-left corner with padding
        NSLayoutConstraint.activate([
            debugLabel.topAnchor.constraint(equalTo: safeAreaLayoutGuide.topAnchor, constant: 16),
            debugLabel.leadingAnchor.constraint(equalTo: safeAreaLayoutGuide.leadingAnchor, constant: 16)
        ])
    }

    // MARK: - Gesture Handlers

    ///  MODAL INPUT: TAP (Finger Only - Select/Deselect Cards)
    @objc func handleTap(_ gesture: UITapGestureRecognizer) {
        let loc = gesture.location(in: self)
        guard let coord = coordinator else { return }

        // 1. Calculate World Point
        let worldPoint = screenToWorldPixels_PureDouble(
            loc,
            viewSize: bounds.size,
            panOffset: coord.panOffset,
            zoomScale: coord.zoomScale,
            rotationAngle: coord.rotationAngle
        )

        // 2. Hit Test Cards (Reverse order = Top first)
        for card in coord.activeFrame.cards.reversed() {
            if card.hitTest(pointInFrame: worldPoint) {
                // Toggle Edit Mode
                card.isEditing.toggle()
                print(" Card \(card.id) Editing: \(card.isEditing)")
                return
            }
        }

        // If we tapped nothing, deselect all cards
        for card in coord.activeFrame.cards {
            card.isEditing = false
        }
        print(" All cards deselected")
    }

    ///  MODAL INPUT: PAN (Finger Only - Drag Card or Pan Canvas)
    @objc func handlePan(_ gesture: UIPanGestureRecognizer) {
        let loc = gesture.location(in: self)
        guard let coord = coordinator else { return }

        // Track which card we are dragging (stored in TouchableMTKView)
        var draggedCard: Card? {
            get { objc_getAssociatedObject(self, &AssociatedKeys.draggedCard) as? Card }
            set { objc_setAssociatedObject(self, &AssociatedKeys.draggedCard, newValue, .OBJC_ASSOCIATION_RETAIN) }
        }

        switch gesture.state {
        case .began:
            // 1. Check if we hit an EDITING card
            let worldPoint = screenToWorldPixels_PureDouble(
                loc,
                viewSize: bounds.size,
                panOffset: coord.panOffset,
                zoomScale: coord.zoomScale,
                rotationAngle: coord.rotationAngle
            )

            for card in coord.activeFrame.cards.reversed() {
                if card.hitTest(pointInFrame: worldPoint) {
                    if card.isEditing {
                        draggedCard = card
                        print(" Started dragging card \(card.id)")
                        return // Stop processing; we are dragging a card
                    }
                    // If card is NOT editing, we ignore it (Pass through to Canvas Pan)
                }
            }
            draggedCard = nil // We are panning the canvas

        case .changed:
            let translation = gesture.translation(in: self)

            if let card = draggedCard {
                //  DRAG CARD
                // Convert screen translation to World translation
                // dxWorld = dxScreen / zoom
                let dx = Double(translation.x) / coord.zoomScale
                let dy = Double(translation.y) / coord.zoomScale

                // Handle rotation of the drag vector if camera is rotated
                let ang = Double(coord.rotationAngle)
                let c = cos(ang), s = sin(ang)
                let dxRot = dx * c + dy * s
                let dyRot = -dx * s + dy * c

                // Update Card Position
                card.origin.x += dxRot
                card.origin.y += dyRot

                // Reset gesture so we get incremental updates
                gesture.setTranslation(.zero, in: self)

            } else {
                //  PAN CANVAS
                // Convert screen translation to world translation (accounting for zoom)
                let dx = Double(translation.x) / coord.zoomScale
                let dy = Double(translation.y) / coord.zoomScale

                // Handle rotation
                let ang = Double(coord.rotationAngle)
                let c = cos(ang), s = sin(ang)
                let dxRot = dx * c + dy * s
                let dyRot = -dx * s + dy * c

                // Update pan offset
                coord.panOffset.x += dxRot
                coord.panOffset.y += dyRot

                gesture.setTranslation(.zero, in: self)
            }

        case .ended, .cancelled, .failed:
            if let card = draggedCard {
                print(" Finished dragging card \(card.id)")
            }
            draggedCard = nil

        default:
            break
        }
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

            //  Normal incremental zoom - multiply using Double precision
            coord.zoomScale = coord.zoomScale * Double(gesture.scale)
            gesture.scale = 1.0

            //  COMMIT 3: TELESCOPING TRANSITIONS
            // Check if we need to drill down (zoom in) or pop up (zoom out)
            // Pass the current touch centroid to anchor transitions to finger position
            //  FIX: If we switched frames, STOP here.
            // The transition logic has already calculated the perfect panOffset
            // to keep the finger pinned. Running the standard solver below
            // would overwrite it with a "glitched" value.
            if checkTelescopingTransitions(coord: coord, currentCentroid: loc) {
                return
            }

            // Standard Solver (Only runs if we did NOT switch frames)
            let target = (activeOwner == .pinch) ? loc : anchorScreen
            coord.panOffset = solvePanOffsetForAnchor_Double(anchorWorld: anchorWorld,
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

            //  Keep shared anchor pinned - use Double-precision solver
            let target = (activeOwner == .rotation) ? loc : anchorScreen
            coord.panOffset = solvePanOffsetForAnchor_Double(anchorWorld: anchorWorld,
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

    @objc func handleLongPress(_ gesture: UILongPressGestureRecognizer) {
        guard let coord = coordinator else { return }

        if gesture.state == .began {
            coord.tileManager.debugMode.toggle()
            let status = coord.tileManager.debugMode ? "ON" : "OFF"
            print("\n TILE DEBUG MODE: \(status)")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            if coord.tileManager.debugMode {
                let loc = gesture.location(in: self)
                let worldPoint = screenToWorldPixels(loc,
                                                     viewSize: bounds.size,
                                                     panOffset: SIMD2<Float>(Float(coord.panOffset.x), Float(coord.panOffset.y)),
                                                     zoomScale: Float(coord.zoomScale),
                                                     rotationAngle: coord.rotationAngle)
                let tileKey = coord.tileManager.getTileKey(worldPoint: worldPoint)
                let debugOutput = coord.tileManager.debugInfo(worldPoint: worldPoint,
                                                              screenPoint: loc,
                                                              tileKey: tileKey)
                print(debugOutput)

                // PHASE 2 TEST: Tile-local tessellation
                coord.testTileLocalTessellation()

                // PHASE 3 DIAGNOSIS: Check for gaps issue
                coord.diagnoseGapsIssue()
            }
        }
    }




    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard event?.allTouches?.count == 1, let touch = touches.first else { return }
        let location = touch.location(in: self)
        coordinator?.handleTouchBegan(at: location, touchType: touch.type)
    }
    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard event?.allTouches?.count == 1, let touch = touches.first else { return }
        let location = touch.location(in: self)

        //  GET PREDICTED TOUCHES
        // These are high-precision estimates of where the finger will be next frame.
        var predictedPoints: [CGPoint] = []

        if let predicted = event?.predictedTouches(for: touch) {
            for pTouch in predicted {
                predictedPoints.append(pTouch.location(in: self))
            }
        }

        // Pass both Real and Predicted to Coordinator
        coordinator?.handleTouchMoved(at: location,
                                    predicted: predictedPoints,
                                    touchType: touch.type)
    }
    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard event?.allTouches?.count == 1, let touch = touches.first else { return }
        let location = touch.location(in: self)
        coordinator?.handleTouchEnded(at: location, touchType: touch.type)
    }
    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard event?.allTouches?.count == 1, let touch = touches.first else { return }
        coordinator?.handleTouchCancelled(touchType: touch.type)
    }
}
