using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using UnityEngine.InputSystem;
using RosMessageTypes.Geometry;
using RosMessageTypes.Tf2;
using RosMessageTypes.Std;
using Unity.Robotics.ROSTCPConnector.ROSGeometry;
using TMPro;
using UnityEngine.XR;

public class HeadsetPublisher : MonoBehaviour
{
    public string unityFrame = "vr_origin";
    public string headsetFrame = "headset";
    public string handFrameLeft = "hand_left";
    public string poseTopic = "/quest/pose";
    public TextMeshProUGUI decimatorText;

    // Kept so your inspector doesn't break, but poses are NOT read from these anymore
    public InputActionReference headsetPose;
    public InputActionReference headsetRotation;
    public InputActionReference handPoseLeft;
    public InputActionReference handRotationLeft;
    public InputActionReference handPoseRight;
    public InputActionReference handRotationRight;

    private Transform root;
    private string handFrameRight = "hand_right";
    private ROSConnection ros;
    private TFMessageMsg tfMsg;
    private PoseStampedMsg headsetPoseMsg;
    private PoseStampedMsg leftHandMsg;
    private PoseStampedMsg rightHandMsg;

    private string rootFrame;
    private int _decimator = 1;
    private int _frameCounter = 0;

    // --- Path A: read tracking from XR subsystem ---
    private static readonly List<XRNodeState> _nodeStates = new List<XRNodeState>(16);

    bool TryGetNodePose(XRNode node, out Vector3 pos, out Quaternion rot)
    {
        pos = Vector3.zero;
        rot = Quaternion.identity;

        _nodeStates.Clear();
        InputTracking.GetNodeStates(_nodeStates);

        bool any = false;
        for (int i = 0; i < _nodeStates.Count; i++)
        {
            var s = _nodeStates[i];
            if (s.nodeType != node) continue;

            bool gotPos = s.TryGetPosition(out pos);
            bool gotRot = s.TryGetRotation(out rot);

            any = gotPos || gotRot;
            break;
        }

        // Unity sometimes returns (0,0,0,0) when invalid
        if (rot == default) rot = Quaternion.identity;

        return any;
    }

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();

        void Awake() { Debug.Log("HeadsetPublisher AWAKE", this); }
        void OnEnable() { Debug.Log("HeadsetPublisher ENABLED", this); }

        handFrameRight = handFrameLeft.Replace("left", "right");

        // Find root safely
        var rootObj = GameObject.FindWithTag("root");
        if (rootObj == null)
        {
            Debug.LogError("Root GameObject with tag 'root' not found. Using rootFrame='odom'.");
            root = null;
            rootFrame = "odom";
        }
        else
        {
            root = rootObj.transform;
            var tfAtt = root.GetComponent<TFAttachment>();
            rootFrame = (tfAtt != null && !string.IsNullOrEmpty(tfAtt.FrameID)) ? tfAtt.FrameID : "odom";
        }

        // Register publishers
        ros.RegisterPublisher<PoseStampedMsg>(poseTopic + "/headset");
        ros.RegisterPublisher<TFMessageMsg>("/tf");

        // (Optional) keep enabling actions if other scripts rely on them.
        // Pose reading in this script does NOT depend on these anymore.
        if (headsetPose != null) headsetPose.action.Enable();
        if (headsetRotation != null) headsetRotation.action.Enable();
        if (handPoseLeft != null) handPoseLeft.action.Enable();
        if (handRotationLeft != null) handRotationLeft.action.Enable();
        if (handPoseRight != null) handPoseRight.action.Enable();
        if (handRotationRight != null) handRotationRight.action.Enable();

        // Init PoseStamped messages (ensure nested fields exist)
        headsetPoseMsg = new PoseStampedMsg();
        headsetPoseMsg.header = new HeaderMsg();
        headsetPoseMsg.pose = new PoseMsg();
        headsetPoseMsg.pose.position = new PointMsg();
        headsetPoseMsg.pose.orientation = new QuaternionMsg();

        // (Optional placeholders)
        leftHandMsg = new PoseStampedMsg();
        leftHandMsg.header = new HeaderMsg();
        leftHandMsg.pose = new PoseMsg();
        leftHandMsg.pose.position = new PointMsg();
        leftHandMsg.pose.orientation = new QuaternionMsg();

        rightHandMsg = new PoseStampedMsg();
        rightHandMsg.header = new HeaderMsg();
        rightHandMsg.pose = new PoseMsg();
        rightHandMsg.pose.position = new PointMsg();
        rightHandMsg.pose.orientation = new QuaternionMsg();

        // Init TF message with 4 transforms and allocate nested fields once
        tfMsg = new TFMessageMsg();
        tfMsg.transforms = new TransformStampedMsg[4];
        for (int i = 0; i < 4; i++)
        {
            tfMsg.transforms[i] = new TransformStampedMsg();
            tfMsg.transforms[i].header = new HeaderMsg();
            tfMsg.transforms[i].transform = new TransformMsg();
            tfMsg.transforms[i].transform.translation = new Vector3Msg();
            tfMsg.transforms[i].transform.rotation = new QuaternionMsg();
        }

        // Set constant TF frame names once
        tfMsg.transforms[0].header.frame_id = rootFrame;
        tfMsg.transforms[0].child_frame_id = unityFrame;

        tfMsg.transforms[1].header.frame_id = unityFrame;
        tfMsg.transforms[1].child_frame_id = headsetFrame;

        tfMsg.transforms[2].header.frame_id = unityFrame;
        tfMsg.transforms[2].child_frame_id = handFrameLeft;

        tfMsg.transforms[3].header.frame_id = unityFrame;
        tfMsg.transforms[3].child_frame_id = handFrameRight;

        if (decimatorText != null)
            decimatorText.text = "TF Decimator: " + _decimator;
    }

    void Update()
    {
        var displays = new List<XRDisplaySubsystem>();
        SubsystemManager.GetInstances(displays);
        if (Time.frameCount % 120 == 0)
        {
            Debug.Log($"XRDisplaySubsystems={displays.Count} XRSettings.enabled={UnityEngine.XR.XRSettings.enabled} loadedDevice={UnityEngine.XR.XRSettings.loadedDeviceName}");
        }

        _frameCounter++;
        if (_decimator < 1) _decimator = 1;
        if (_frameCounter % _decimator != 0)
            return;
        _frameCounter = 0;

        // --- Path A pose acquisition ---
        bool headOk = TryGetNodePose(XRNode.Head, out Vector3 headPosU, out Quaternion headRotU);
        bool leftOk = TryGetNodePose(XRNode.LeftHand, out Vector3 leftPosU, out Quaternion leftRotU);
        bool rightOk = TryGetNodePose(XRNode.RightHand, out Vector3 rightPosU, out Quaternion rightRotU);

        if (!headOk && !leftOk && !rightOk)
        {
            if (Time.frameCount % 120 == 0)
                Debug.LogWarning("HeadsetPublisher: XR nodes not providing pose yet (tracking not ready / headset not worn / controllers asleep?).");
        }

        // Convert to ROS FLU messages
        Vector3Msg headPos = headPosU.To<FLU>();
        QuaternionMsg headRot = headRotU.To<FLU>();

        Vector3Msg leftPos = leftPosU.To<FLU>();
        QuaternionMsg leftRot = leftRotU.To<FLU>();

        Vector3Msg rightPos = rightPosU.To<FLU>();
        QuaternionMsg rightRot = rightRotU.To<FLU>();

        // Root -> unityFrame transform
        if (root != null)
        {
            tfMsg.transforms[0].transform.translation = root.InverseTransformPoint(Vector3.zero).To<FLU>();
            tfMsg.transforms[0].transform.rotation = Quaternion.Inverse(root.rotation).To<FLU>();
        }
        else
        {
            tfMsg.transforms[0].transform.translation.x = 0;
            tfMsg.transforms[0].transform.translation.y = 0;
            tfMsg.transforms[0].transform.translation.z = 0;
            tfMsg.transforms[0].transform.rotation.x = 0;
            tfMsg.transforms[0].transform.rotation.y = 0;
            tfMsg.transforms[0].transform.rotation.z = 0;
            tfMsg.transforms[0].transform.rotation.w = 1;
        }

        // unityFrame -> headset/left/right
        tfMsg.transforms[1].transform.translation = headPos;
        tfMsg.transforms[1].transform.rotation = headRot;

        tfMsg.transforms[2].transform.translation = leftPos;
        tfMsg.transforms[2].transform.rotation = leftRot;

        tfMsg.transforms[3].transform.translation = rightPos;
        tfMsg.transforms[3].transform.rotation = rightRot;

        // PoseStamped /quest/pose/headset (PointMsg!)
        headsetPoseMsg.header.frame_id = unityFrame;

        headsetPoseMsg.pose.position.x = headPos.x;
        headsetPoseMsg.pose.position.y = headPos.y;
        headsetPoseMsg.pose.position.z = headPos.z;

        headsetPoseMsg.pose.orientation = headRot;

        ros.Publish(poseTopic + "/headset", headsetPoseMsg);
        ros.Publish("/tf", tfMsg);
    }

    public void OnDecimatorChange(float value)
    {
        _decimator = (int)value;
        if (decimatorText != null)
            decimatorText.text = "TF Decimator: " + _decimator;
    }
}
