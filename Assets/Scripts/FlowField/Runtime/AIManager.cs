using System;
using System.Collections.Generic;
using Cysharp.Threading.Tasks;
using Unity.VisualScripting;
using UnityEngine;
using Random = UnityEngine.Random;

public class AIManager : MonoBehaviour
{
    [SerializeField]
    private AIEntity AIEntityOriginal;

    [SerializeField]
    private int AICount = 5;

    private List<AIEntity> AIEntities = new();

    [SerializeField]
    private MapGenerator Map;

    [SerializeField]
    private FlowFieldWayFinder wayFinder;

    private bool IsTargetWatching = false;

    private void Start()
    {
        for (int i = 0; i < AICount; i++)
        {
            var entity = GameObject.Instantiate<AIEntity>(AIEntityOriginal, transform);

            entity.Setup(this);
            AIEntities.Add(entity);
        }

        RandomEntityPosition();
    }

    private void RandomEntityPosition()
    {
        for (int i = 0; i < AICount; i++)
        {
            while (true)
            {
                var randomX = Random.Range(0, Map.MapSize.x);
                var randomY = Random.Range(0, Map.MapSize.y);
                var initPosition = new Vector3(randomX + Random.Range(-0.5f, 0.5f), 0.5f, randomY + Random.Range(-0.5f, 0.5f));
                var block = Map.GetPlacedBlock(initPosition);
                if (Map.FlowFieldBlocks[block.x, block.y].BlockType == BlockType.Normal)
                {
                    AIEntities[i].transform.position = initPosition;
                    break;
                }
            }

        }
    }

    private void OnGUI()
    {
        if (GUILayout.Button("RandomPosition", GUILayout.Width(200), GUILayout.Height(100)))
        {
            RandomEntityPosition();
        }

        if (GUILayout.Button("Navigate", GUILayout.Width(200), GUILayout.Height(100)))
        {
            AIEntities.ForEach(entity => entity.SetNavigate(true));
        }

        IsTargetWatching = GUILayout.Toggle(IsTargetWatching, "Target Watching", GUILayout.Width(200), GUILayout.Height(100));

        if (GUILayout.Button("Index", GUILayout.Width(200), GUILayout.Height(100)))
        {
            // ShowContent
            for (int i = 0; i < Map.MapSize.x; i++)
            {
                for (int j = 0; j < Map.MapSize.y; j++)
                {
                    Map.FlowFieldBlocks[i, j].ShowContent(ContentType.Index);
                }
            }
        }

        if (GUILayout.Button("HeatMap", GUILayout.Width(200), GUILayout.Height(100)))
        {
            for (int i = 0; i < Map.MapSize.x; i++)
            {
                for (int j = 0; j < Map.MapSize.y; j++)
                {
                    Map.FlowFieldBlocks[i, j].ShowContent(ContentType.HeatMap);
                }
            }
        }

        if (GUILayout.Button("Direction", GUILayout.Width(200), GUILayout.Height(100)))
        {
            for (int i = 0; i < Map.MapSize.x; i++)
            {
                for (int j = 0; j < Map.MapSize.y; j++)
                {
                    Map.FlowFieldBlocks[i, j].ShowContent(ContentType.Direction);
                }
            }
        }

        // IsDebugFindCorner = GUILayout.Toggle(IsDebugFindCorner, "Debug Find corner");
    }

    private void Update()
    {
        if (IsTargetWatching)
        {
            var y = Camera.main.transform.position.y;
            Camera.main.transform.position = new Vector3(AIEntities[0].transform.position.x, y, AIEntities[0].transform.position.z);
        }
    }

    public Vector3 GetMoveDirection(AIEntity entity)
    {
        // Map.GetPlacedBlock(position);
        var position = entity.transform.position;

        var positionList = entity.SamplePositionList;

        var clostestCornerPosition = Map.GetClostestCorner(position);

        // var z0 = positionList[0].z - clostestCornerPosition.z;
        var xList = new List<float>();
        var zList = new List<float>();
        var directionList = new List<Vector3>();

        for (int i = 0; i < positionList.Count; i++)
        {
            var blockIdx = Map.GetPlacedBlock(positionList[i]);
            // (blockIdx.y, blockIdx.x) = (blockIdx.x, blockIdx.y);
            var direction = Vector3.zero;
            var isValidBlock = false;
            if (Utils.CheckValidPosition(blockIdx, Map.MapSize))
            {
                var block = Map.FlowFieldBlocks[blockIdx.x, blockIdx.y];
                if (block.BlockType != BlockType.Obstacle)
                {
                    isValidBlock = true;
                    direction = block.Direction;
                }
            }

            if (isValidBlock)
            {
                xList.Add(Mathf.Abs(position.x - clostestCornerPosition.x));
                zList.Add(Mathf.Abs(position.z - clostestCornerPosition.z));
            }
            else
            {
                xList.Add(0);
                zList.Add(0);
            }
            directionList.Add(new(direction.x, 0, direction.y));
        }
        var d01 = Vector3.zero;
        var d23 = Vector3.zero;
        var final = Vector3.zero;
        if ((zList[0] + zList[1]) != 0)
            d01 = (directionList[0] * zList[0] + directionList[1] * zList[1]) / (zList[0] + zList[1]);

        if ((zList[2] + zList[3]) != 0)
            d23 = (directionList[2] * zList[2] + directionList[3] * zList[3]) / (zList[2] + zList[3]);

        if ((xList[0] + xList[1] + xList[2] + xList[3]) != 0)
            final = (d01 * (xList[0] + xList[1]) + d23 * (xList[2] + xList[3])) / (xList[0] + xList[1] + xList[2] + xList[3]);

        return final.normalized;
    }

    public bool IsReach(Vector3 position)
    {
        var blockIdx = Map.GetPlacedBlock(position);
        (blockIdx.y, blockIdx.x) = (blockIdx.x, blockIdx.y);
        if (Utils.CheckValidPosition(blockIdx, Map.MapSize))
            return Map.FlowFieldBlocks[blockIdx.x, blockIdx.y].BlockType == BlockType.End;
        return false;
    }

}