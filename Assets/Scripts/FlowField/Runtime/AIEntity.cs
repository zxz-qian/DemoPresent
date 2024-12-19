using System;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(BoxCollider))]
public class AIEntity : MonoBehaviour
{
    private AIManager AIManager;

    [SerializeField]
    private float Speed = 10;

    [SerializeField]
    private bool Navigating = false;

    private List<Vector3> samplePositionList = new(4);
    public List<Vector3> SamplePositionList => samplePositionList;

    private BoxCollider boxCollider;

    private List<GameObject> DebugRouteGOs = new();

    [SerializeField]
    private GameObject DebugSphereOrignial;

    private const int UPDATE_DIR_FRAME_INTERVAL = 10;
    private int updateDirFrameCount = 0;

    private void Start()
    {
        boxCollider = GetComponent<BoxCollider>();
        for (int i = 0; i < 4; i++)
        {
            samplePositionList.Add(Vector3.zero);
        }
    }

    public void Setup(AIManager AIManager)
    {
        this.AIManager = AIManager;
    }

    public void SetNavigate(bool isNavigate)
    {
        Navigating = isNavigate;
        DebugRouteGOs.ForEach(go => Destroy(go));
        DebugRouteGOs.Clear();
    }

    private void Update()
    {
        if (Navigating)
        {
            UpdateSamplePositionList();
            if (AIManager.IsReach(transform.position))
                Navigating = false;
            Vector3 dir = AIManager.GetMoveDirection(this);
            if (dir.sqrMagnitude < float.Epsilon)
                return;

            if (updateDirFrameCount >= UPDATE_DIR_FRAME_INTERVAL)
            {
                transform.forward = dir;
                updateDirFrameCount = 0;
            }
            updateDirFrameCount++;

            transform.position += transform.forward * Time.deltaTime * Speed;
            // transform.

            var sphere = GameObject.Instantiate(DebugSphereOrignial, transform.position, Quaternion.identity);
            // sphere.transform.parent = transform;
            DebugRouteGOs.Add(sphere);
        }
    }



    private void UpdateSamplePositionList()
    {
        samplePositionList[0] = transform.position - boxCollider.bounds.extents.x * transform.right - boxCollider.bounds.extents.z * transform.forward;
        samplePositionList[1] = transform.position + boxCollider.bounds.extents.x * transform.right - boxCollider.bounds.extents.z * transform.forward;
        samplePositionList[2] = transform.position + boxCollider.bounds.extents.x * transform.right + boxCollider.bounds.extents.z * transform.forward;
        samplePositionList[3] = transform.position - boxCollider.bounds.extents.x * transform.right + boxCollider.bounds.extents.z * transform.forward;
    }
}