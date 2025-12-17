// TouchableMTKView.swift subclasses MTKView to attach gestures, manage anchors,
// and forward user interactions to the coordinator while presenting debug HUD info.
import UIKit
import MetalKit
import ObjectiveC.runtime
import simd

// MARK: - Associated Object Keys for gesture state storage
private struct AssociatedKeys {
    static var dragContext: UInt8 = 0
}

// MARK: - Drag Context for Cross-Depth Dragging
/// Stores the card being dragged and its coordinate conversion scale
/// The conversion scale translates movement between coordinate systems:
///   - Parent cards: scale < 1.0 (move slower)
///   - Active cards: scale = 1.0 (normal movement)
///   - Child cards: scale > 1.0 (move faster)
private class DragContext {
    let card: Card
    let conversionScale: Double

    init(card: Card, conversionScale: Double) {
        self.card = card
        self.conversionScale = conversionScale
    }
}

// MARK: - TouchableMTKView
class TouchableMTKView: MTKView {
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

    weak var coordinator: Coordinator?

    var panGesture: UIPanGestureRecognizer!
    var pinchGesture: UIPinchGestureRecognizer!
    var rotationGesture: UIRotationGestureRecognizer!
    var longPressGesture: UILongPressGestureRecognizer!
    var cardLongPressGesture: UILongPressGestureRecognizer! // Single-finger long press for card settings

    // Debug HUD
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

    // MARK: - Telescoping Transitions

    /// Check if zoom has exceeded thresholds and perform frame transitions if needed.
    /// Returns TRUE if a transition occurred (caller should return early).
    func checkTelescopingTransitions(coord: Coordinator,
                                     anchorWorld: SIMD2<Double>,
                                     anchorScreen: CGPoint) -> Bool {
        // DRILL DOWN
        if coord.zoomScale > 1000.0 {
            drillDownToNewFrame(coord: coord,
                                anchorWorld: anchorWorld,
                                anchorScreen: anchorScreen)
            return true
        }
        // POP UP (Telescope Out - create parent if needed)
        else if coord.zoomScale < 0.5 {
            popUpToParentFrame(coord: coord,
                               anchorWorld: anchorWorld,
                               anchorScreen: anchorScreen)
            return true
        }

        return false
    }

    /// "The Silent Teleport" - Drill down into a child frame.
    ///  FIX: Uses the shared anchor instead of recomputing to prevent micro-jumps.
    func drillDownToNewFrame(coord: Coordinator,
                             anchorWorld: SIMD2<Double>,
                             anchorScreen: CGPoint) {
        print("🚀 ENTER drillDownToNewFrame at depth \(coord.activeFrame.depthFromRoot), zoom \(coord.zoomScale)")

        // 1. CAPTURE STATE (CRITICAL: Do this BEFORE resetting zoom)
        let currentZoom = coord.zoomScale // This should be ~1000.0

        // 2. USE THE EXACT ANCHOR (Don't recompute from screen!)
        // This is the key fix - we use the exact same world point that the gesture handler
        // has been tracking, preventing floating point discrepancies.
        let pinchPointWorld = anchorWorld // EXACT same world point as gesture anchor
        let currentCentroid = anchorScreen // Reuse for solving pan

        // 3. Check for existing child frame
        // In the telescope chain, there should be exactly ONE child per frame
        // We always re-enter it regardless of distance (telescope chain invariant)
        let currentDepth = coord.activeFrame.depthFromRoot

        print("🔍 Checking for child frames at depth \(currentDepth), \(coord.activeFrame.children.count) children")

        var targetFrame: Frame? = nil

        // TELESCOPE CHAIN INVARIANT: If there's exactly one child, always use it
        // This maintains the linked-list structure (no siblings in telescope chain)
        if coord.activeFrame.children.count == 1 {
            targetFrame = coord.activeFrame.children[0]
            print("  ✓ Found single child at depth \(targetFrame!.depthFromRoot) (telescope chain)")
        } else if coord.activeFrame.children.count > 1 {
            // Multiple children exist (should only happen for non-telescope frames)
            // Use search radius to find the closest one
            let searchRadius: Double = 50.0
            for child in coord.activeFrame.children {
                let dist = distance(child.originInParent, pinchPointWorld)
                print("  - Child at depth \(child.depthFromRoot), origin \(child.originInParent), distance \(dist)")
                if dist < searchRadius {
                    targetFrame = child
                    print("  ✓ Selected this child (within search radius)")
                    break
                }
            }
        }

        if let existing = targetFrame {
            //  RE-ENTER EXISTING FRAME
            let oldDepth = coord.activeFrame.depthFromRoot
            coord.activeFrame = existing
            let newDepth = coord.activeFrame.depthFromRoot

            print("🔭 Re-entered existing frame (Telescope In): depth \(oldDepth) → \(newDepth), zoom \(currentZoom) → will be \(currentZoom / existing.scaleRelativeToParent)")

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

        } else {
            //  CREATE NEW FRAME
            //  This creates a NEW branch (sibling to existing children, if any)
            //  For pure telescoping, this should only happen when there are NO existing children

            //  FIX: Center the new frame exactly on the PINCH POINT (Finger).
            // This prevents exponential coordinate growth (Off-Center Accumulation).
            // OLD: Centered on screen center → 500px offset compounds to 500,000 → 500M → 10^18 → CRASH
            // NEW: Centered on finger → offset resets to 0 at each depth → stays bounded forever
            let newFrameOrigin = pinchPointWorld

            let oldDepth = coord.activeFrame.depthFromRoot
            let newDepth = oldDepth + 1  // Child is always parent + 1

            let newFrame = Frame(
                parent: coord.activeFrame,
                origin: newFrameOrigin,
                scale: currentZoom,  // Use captured high zoom
                depth: newDepth
            )

            coord.activeFrame.children.append(newFrame)
            coord.activeFrame = newFrame
            coord.zoomScale = 1.0

            if coord.activeFrame.parent?.children.count ?? 0 > 1 {
                print("⚠️ WARNING: Created sibling frame (multiple children at depth \(oldDepth))")
            }
            print("🔭 Created new frame (Telescope In): depth \(oldDepth) → \(newDepth), zoom \(currentZoom) → 1.0")

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

        print("🏁 EXIT drillDownToNewFrame at depth \(coord.activeFrame.depthFromRoot), zoom \(coord.zoomScale)")
    }

    /// Helper: Calculate Euclidean distance between two points
    func distance(_ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Double {
        let dx = b.x - a.x
        let dy = b.y - a.y
        return sqrt(dx * dx + dy * dy)
    }

    /// "The Reverse Teleport" - Pop up to the parent frame.
    /// If no parent exists, creates a new "Super Root" containing the current universe.
    ///  FIX: Uses the shared anchor instead of recomputing to prevent micro-jumps.
    func popUpToParentFrame(coord: Coordinator,
                            anchorWorld: SIMD2<Double>,
                            anchorScreen: CGPoint) {
        print("🚀 ENTER popUpToParentFrame at depth \(coord.activeFrame.depthFromRoot), zoom \(coord.zoomScale)")

        let currentFrame = coord.activeFrame

        // CREATE PARENT IF NEEDED (Telescope Out Beyond Root)
        let parent: Frame
        if let existingParent = currentFrame.parent {
            parent = existingParent
        } else {
            // Create new "Super Root" that contains the current universe
            let currentDepth = currentFrame.depthFromRoot
            let newParent = Frame(depth: currentDepth - 1)

            // Link them
            currentFrame.parent = newParent
            newParent.children.append(currentFrame)

            // Position: Center the old universe at (0,0) in the new one
            // FLOATING ORIGIN: This resets coordinates to prevent large values
            // Works for BOTH positive (drill in) and negative (telescope out) depths
            currentFrame.originInParent = .zero

            // Scale: 1000x (symmetric with drill-down)
            currentFrame.scaleRelativeToParent = 1000.0

            parent = newParent

            // Debug logging
            print("🔭 Created new parent frame (Telescope Out): depth \(currentDepth) → \(newParent.depthFromRoot), zoom \(coord.zoomScale) → will be \(coord.zoomScale * currentFrame.scaleRelativeToParent)")
        }

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
        let oldDepth = currentFrame.depthFromRoot
        coord.activeFrame = parent
        coord.zoomScale = newZoom
        coord.panOffset = newPanOffset
        let newDepth = coord.activeFrame.depthFromRoot

        print("🔭 Completed telescope out transition: depth \(oldDepth) → \(newDepth), zoom now \(newZoom)")

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

        print("🏁 EXIT popUpToParentFrame at depth \(coord.activeFrame.depthFromRoot), zoom \(coord.zoomScale)")
    }

    /// Helper: Calculate the depth of a frame relative to root
    /// Returns how far up the tree we've traversed (can be negative if above root)
    func frameDepth(_ frame: Frame) -> Int {
        // Just count total parents for now - positive means "above original root"
        var depth = 0
        var current: Frame? = frame
        while current?.parent != nil {
            depth += 1
            current = current?.parent
        }
        return depth
    }

    /// Helper: Calculate absolute depth (distance from rootFrame)
    /// Positive = drilled down (child of root), Negative = telescoped out (parent of root)
    func relativeDepth(frame: Frame, root: Frame) -> Int {
        if frame === root {
            return 0
        }

        // Check if frame is below root (child)
        var current: Frame? = frame
        var tempDepth = 0
        while let parent = current?.parent {
            tempDepth += 1
            if parent === root {
                return tempDepth  // Positive depth
            }
            current = parent
        }

        // Check if frame is above root (parent)
        current = root
        tempDepth = 0
        while let parent = current?.parent {
            tempDepth -= 1
            if parent === frame {
                return tempDepth  // Negative depth
            }
            current = parent
        }

        return 0  // Shouldn't reach here
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
        // Crucial: Ignore Apple Pencil for panning/dragging.
        // On Mac Catalyst, prefer two-finger trackpad scrolling for panning so click-drag can be used for drawing.
        if isRunningOnMac {
            panGesture.allowedTouchTypes = []
            if #available(iOS 13.4, macCatalyst 13.4, *) {
                panGesture.allowedScrollTypesMask = [.continuous, .discrete]
            }
        } else {
            panGesture.allowedTouchTypes = [NSNumber(value: UITouch.TouchType.direct.rawValue)]
        }
        addGestureRecognizer(panGesture)

        //  MODAL INPUT: TAP (Finger Only - Select/Edit Cards)
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
        if isRunningOnMac {
            var tapTouchTypes: [NSNumber] = [NSNumber(value: UITouch.TouchType.direct.rawValue)]
            if #available(iOS 13.4, macCatalyst 13.4, *) {
                tapTouchTypes.append(NSNumber(value: UITouch.TouchType.indirectPointer.rawValue))
            }
            tapGesture.allowedTouchTypes = tapTouchTypes
        } else {
            tapGesture.allowedTouchTypes = [NSNumber(value: UITouch.TouchType.direct.rawValue)]
        }
        addGestureRecognizer(tapGesture)

        pinchGesture = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
        addGestureRecognizer(pinchGesture)

        rotationGesture = UIRotationGestureRecognizer(target: self, action: #selector(handleRotation(_:)))
        addGestureRecognizer(rotationGesture)

        longPressGesture = UILongPressGestureRecognizer(target: self, action: #selector(handleLongPress(_:)))
        longPressGesture.minimumPressDuration = 0.5
        longPressGesture.numberOfTouchesRequired = 2
        addGestureRecognizer(longPressGesture)

        // Single-finger long press for card settings
        cardLongPressGesture = UILongPressGestureRecognizer(target: self, action: #selector(handleCardLongPress(_:)))
        cardLongPressGesture.minimumPressDuration = 0.5
        cardLongPressGesture.numberOfTouchesRequired = 1
        if isRunningOnMac {
            var longPressTouchTypes: [NSNumber] = [NSNumber(value: UITouch.TouchType.direct.rawValue)]
            if #available(iOS 13.4, macCatalyst 13.4, *) {
                longPressTouchTypes.append(NSNumber(value: UITouch.TouchType.indirectPointer.rawValue))
            }
            cardLongPressGesture.allowedTouchTypes = longPressTouchTypes
        } else {
            cardLongPressGesture.allowedTouchTypes = [NSNumber(value: UITouch.TouchType.direct.rawValue)] // Finger only
        }
        addGestureRecognizer(cardLongPressGesture)

        panGesture.delegate = self
        pinchGesture.delegate = self
        rotationGesture.delegate = self
        longPressGesture.delegate = self
        cardLongPressGesture.delegate = self

        // Setup Debug HUD
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

        // Use the new hierarchical hit test to find cards at any depth
        if let result = coord.hitTestHierarchy(screenPoint: loc, viewSize: bounds.size) {
            // Toggle Edit on the card (wherever it lives - parent, active, or child)
            result.card.isEditing.toggle()
            return
        }

        // If we tapped nothing, deselect all cards (requires recursive clear)
        clearSelectionRecursive(frame: coord.rootFrame)
    }

    /// Helper to clear all card selections recursively across the entire hierarchy
    func clearSelectionRecursive(frame: Frame) {
        frame.cards.forEach { $0.isEditing = false }
        frame.children.forEach { clearSelectionRecursive(frame: $0) }
    }

    ///  MODAL INPUT: PAN (Finger Only - Drag Card or Pan Canvas)
    /// Now supports cross-depth dragging with proper coordinate conversion
    @objc func handlePan(_ gesture: UIPanGestureRecognizer) {
        let loc = gesture.location(in: self)
        guard let coord = coordinator else { return }

        // Track drag context (card + conversion scale)
        var dragContext: DragContext? {
            get { objc_getAssociatedObject(self, &AssociatedKeys.dragContext) as? DragContext }
            set { objc_setAssociatedObject(self, &AssociatedKeys.dragContext, newValue, .OBJC_ASSOCIATION_RETAIN) }
        }

        switch gesture.state {
        case .began:
            // Hit test hierarchy to find card AND its coordinate scale
            if let result = coord.hitTestHierarchy(screenPoint: loc, viewSize: bounds.size) {
                if result.card.isEditing {
                    // Store Card + Scale Factor for cross-depth dragging
                    dragContext = DragContext(card: result.card, conversionScale: result.conversionScale)
                    return
                }
            }
            dragContext = nil // Pan Canvas

	        case .changed:
            let translation = gesture.translation(in: self)

            if let context = dragContext {
                //  DRAG CARD (Cross-Depth Compatible)

                // 1. Convert Screen Delta -> Active World Delta
                let dxActive = Double(translation.x) / coord.zoomScale
                let dyActive = Double(translation.y) / coord.zoomScale

                // 2. Apply Camera Rotation
                let ang = Double(coord.rotationAngle)
                let c = cos(ang), s = sin(ang)
                let dxRot = dxActive * c + dyActive * s
                let dyRot = -dxActive * s + dyActive * c

                // 3. Convert Active Delta -> Target Frame Delta
                // Use the conversion scale we found during Hit Test!
                // Parent: Scale < 1.0 (Move slower - parent coords are smaller)
                // Child: Scale > 1.0 (Move faster - child coords are larger)
                context.card.origin.x += dxRot * context.conversionScale
                context.card.origin.y += dyRot * context.conversionScale

                gesture.setTranslation(.zero, in: self)

            } else {
                //  PAN CANVAS (Existing Logic)
                let dx = Double(translation.x) / coord.zoomScale
                let dy = Double(translation.y) / coord.zoomScale

                let ang = Double(coord.rotationAngle)
                let c = cos(ang), s = sin(ang)

                coord.panOffset.x += dx * c + dy * s
                coord.panOffset.y += -dx * s + dy * c
                gesture.setTranslation(.zero, in: self)
            }

        case .ended, .cancelled, .failed:
            dragContext = nil

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
            // Start of pinch: claim the anchor if nobody owns it yet.
            lastPinchTouchCount = tc
            if activeOwner == .none {
                lockAnchor(owner: .pinch, at: loc, coord: coord)
            }

        case .changed:
            // If finger count changes, re-lock to new centroid WITHOUT moving content.
            if activeOwner == .pinch, tc != lastPinchTouchCount {
                relockAnchorAtCurrentCentroid(owner: .pinch,
                                              screenPt: loc,
                                              coord: coord)
                lastPinchTouchCount = tc
                // Avoid solving pan on this frame; we just re-synced the anchor.
                gesture.scale = 1.0
                return
            }

	            // Normal incremental zoom (always relative).
	            //
	            // On Mac (Catalyst / iOS app on Mac), trackpad pinch deltas are much more aggressive than iPad,
	            // so we dampen the gesture scale to slow zoom down.
	            let rawScale = max(Double(gesture.scale), 1e-6)
	            let appliedScale: Double
	            if isRunningOnMac {
	                let macZoomSensitivity = 0.25
	                appliedScale = pow(rawScale, macZoomSensitivity)
	            } else {
	                appliedScale = rawScale
	            }
	            coord.zoomScale *= appliedScale
	            gesture.scale = 1.0

            // 🔑 IMPORTANT: depth switches are driven by the *shared anchor*,
            // not by re-sampling world from the current centroid.
            if checkTelescopingTransitions(coord: coord,
                                           anchorWorld: anchorWorld,
                                           anchorScreen: anchorScreen) {
                // The transition functions already solved a perfect panOffset
                // to keep the anchor pinned. Do NOT overwrite it this frame.
                return
            }

            // No frame switch: solve panOffset to keep the anchor fixed.
            let targetScreen: CGPoint = (activeOwner == .pinch) ? loc : anchorScreen

            coord.panOffset = solvePanOffsetForAnchor_Double(
                anchorWorld: anchorWorld,
                desiredScreen: targetScreen,
                viewSize: bounds.size,
                zoomScale: coord.zoomScale,
                rotationAngle: coord.rotationAngle
            )

            if activeOwner == .pinch {
                anchorScreen = targetScreen
            }

        case .ended, .cancelled, .failed:
            if activeOwner == .pinch {
                // If rotation is active, hand off the anchor smoothly.
                if rotationGesture.state == .changed || rotationGesture.state == .began {
                    let rloc = rotationGesture.location(in: self)
                    handoffAnchor(to: .rotation, screenPt: rloc, coord: coord)
                } else {
                    clearAnchorIfUnused()
                }
            }
            lastPinchTouchCount = 0

        default:
            break
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
        // Two-finger long press - currently unused
        // Could be used for global actions (zoom reset, frame navigation, etc.)
    }

    /// Single-finger long press to open card settings
    @objc func handleCardLongPress(_ gesture: UILongPressGestureRecognizer) {
        guard let coord = coordinator else { return }

        if gesture.state == .began {
            let location = gesture.location(in: self)
            coord.handleLongPress(at: location)
        }
    }




    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first else { return }
        if !isRunningOnMac {
            guard event?.allTouches?.count == 1 else { return }
        }
        let location = touch.location(in: self)
        coordinator?.handleTouchBegan(at: location, touchType: touch.type)
    }
    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first else { return }
        if !isRunningOnMac {
            guard event?.allTouches?.count == 1 else { return }
        }
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
        guard let touch = touches.first else { return }
        if !isRunningOnMac {
            guard event?.allTouches?.count == 1 else { return }
        }
        let location = touch.location(in: self)
        coordinator?.handleTouchEnded(at: location, touchType: touch.type)
    }
    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first else { return }
        if !isRunningOnMac {
            guard event?.allTouches?.count == 1 else { return }
        }
        coordinator?.handleTouchCancelled(touchType: touch.type)
    }
}
