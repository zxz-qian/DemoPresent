using System;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

public enum BlockType
{
    Start = 0,
    End = 1,
    Obstacle = 2,
    Normal = 3
}

public class MapGenerator : MonoBehaviour
{
    [SerializeField]
    private FlowFieldBlock blockOriginal;


    [SerializeField]
    private Vector2Int mapSize = new(10, 10);
    [SerializeField]
    private Vector2Int startPosition = new(1, 1);
    [SerializeField]
    private Vector2Int endPosition = new(8, 8);
    [SerializeField]
    private List<Vector2Int> BlockPositions = new()
    {
        new Vector2Int(4, 1),
        new Vector2Int(4, 2),
        new Vector2Int(4, 3),
        new Vector2Int(4, 4),
        new Vector2Int(4, 5),
    };

    [SerializeField]
    private Vector2 blockSize = new(1.0f, 1.0f);

    public FlowFieldBlock[,] FlowFieldBlocks => flowFieldBlocks;
    public Vector2Int MapSize => mapSize;
    public Vector2Int EndPosition => endPosition;
    public Vector2Int StartPosition => startPosition;
    public Vector2 BlockSize => blockSize;

    private FlowFieldBlock[,] flowFieldBlocks;

    [SerializeField]
    private Material StartBlockMaterial;
    [SerializeField]
    private Material EndBlockMaterial;
    [SerializeField]
    private Material ObstacleBlockMaterial;
    [SerializeField]
    private Material NormalBlockMaterial;
    // Start is called before the first frame update
    void Start()
    {
        // MakeMap();
    }

    private void Awake()
    {
        SearchMap();
    }

    public void MakeMap()
    {
        int count = 0;
        flowFieldBlocks = new FlowFieldBlock[mapSize.x, mapSize.y];
        for (int i = 0; i < mapSize.y; i++)
        {
            for (int j = 0; j < mapSize.x; j++)
            {
                flowFieldBlocks[i, j] = GameObject.Instantiate<FlowFieldBlock>(blockOriginal, transform);
                flowFieldBlocks[i, j].transform.position = new Vector3(j, 0, i);
                flowFieldBlocks[i, j].Setup(this, count);
                flowFieldBlocks[i, j].SetBlockType(BlockType.Normal);
                count++;
            }
        }


        if (Utils.CheckValidPosition(startPosition, mapSize))
        {
            flowFieldBlocks[startPosition.x, startPosition.y].SetBlockType(BlockType.Start);
        }

        if (Utils.CheckValidPosition(endPosition, mapSize))
        {
            flowFieldBlocks[endPosition.x, endPosition.y].SetBlockType(BlockType.End);
        }

        for (int i = 0; i < BlockPositions.Count; i++)
        {
            if (Utils.CheckValidPosition(BlockPositions[i], mapSize))
            {
                flowFieldBlocks[BlockPositions[i].x, BlockPositions[i].y].SetBlockType(BlockType.Obstacle);
            }
        }
    }

    private void SearchMap()
    {
        int count = 0;
        flowFieldBlocks = new FlowFieldBlock[mapSize.x, mapSize.y];

        for (int i = 0; i < mapSize.y; i++)
        {
            for (int j = 0; j < mapSize.x; j++)
            {
                flowFieldBlocks[j, i] = transform.GetChild(count).GetComponent<FlowFieldBlock>();
                flowFieldBlocks[j, i].Setup(this, count);

                if (flowFieldBlocks[j, i].BlockType == BlockType.End)
                    endPosition = new(j, i);
                count++;
            }
        }
    }

    private bool IsDebugFindCorner = false;

    private void OnGUI()
    {

    }

    private void OnDrawGizmos()
    {
        if (IsDebugFindCorner)
        {
            if (Selection.activeGameObject != null)
            {
                var corner = GetClostestCorner(Selection.activeGameObject.transform.position);
                Gizmos.DrawSphere(corner, 0.1f);
            }
        }
    }

    public Material GetMaterial(BlockType blockType)
    {
        return blockType switch
        {
            BlockType.Start => StartBlockMaterial,
            BlockType.End => EndBlockMaterial,
            BlockType.Obstacle => ObstacleBlockMaterial,
            BlockType.Normal => NormalBlockMaterial,
            _ => throw new System.InvalidOperationException()
        };
    }


    public Vector3 GetClostestCorner(Vector3 position)
    {
        var blockIdx = GetPlacedBlock(position);
        var blockCenter = new Vector3(BlockSize.x * blockIdx.x, 0, BlockSize.y * blockIdx.y);
        var offset = position - blockCenter;
        return blockCenter + new Vector3(BlockSize.x / 2 * Mathf.Sign(offset.x), 0, BlockSize.y / 2 * Mathf.Sign(offset.z));
    }

    public Vector2Int GetPlacedBlock(Vector3 position)
    {
        var x = (int)(position.x / BlockSize.x);
        var z = (int)(position.z / BlockSize.y);
        return new(x, z);
    }

}
