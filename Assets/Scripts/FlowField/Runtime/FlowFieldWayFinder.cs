using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Cysharp.Threading.Tasks;
public class FlowFieldWayFinder : MonoBehaviour
{
    [SerializeField]
    private MapGenerator Map;

    [SerializeField]
    private bool DemoMode;
    // Start is called before the first frame update
    void Start()
    {
        CreateFlowFieldMap();
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.A))
        {
            MarkBlocksDistance();
        }

        if (Input.GetKeyDown(KeyCode.S))
        {
            MarkBlockDirection();
        }
    }

    private void MarkBlocksDistance()
    {
        MarkBlocksDistanceAsync().Forget();
    }

    private async UniTask MarkBlocksDistanceAsync()
    {
        await UniTask.Yield();
        var blocks = Map.FlowFieldBlocks;
        FlowFieldBlock end = blocks[Map.EndPosition.x, Map.EndPosition.y];
        end.SetHeatValue(0);
        var queue = new Queue<FlowFieldBlock>();
        queue.Enqueue(end);

        while (queue.Count > 0)
        {
            var cur = queue.Dequeue();
            var heatValue = cur.HeatValue;
            var row = cur.Index % Map.MapSize.x;
            var column = cur.Index / Map.MapSize.y;
            var adjacents = Utils.GetAdjacents(new(row, column));

            for (int i = 0; i < adjacents.Count; i++)
            {
                if (Utils.CheckValidPosition(adjacents[i], Map.MapSize))
                {
                    var item = blocks[adjacents[i].x, adjacents[i].y];

                    if (item.HeatValue < 0 && item.BlockType != BlockType.Obstacle)
                    {
                        item.SetHeatValue(heatValue + 1);
                        queue.Enqueue(item);
                    }
                }
            }
            if (DemoMode)
                await UniTask.Delay(250);
        }
    }

    private void MarkBlockDirection()
    {
        MarkBlocksDirectionAsync().Forget();
    }

    private async UniTask MarkBlocksDirectionAsync()
    {
        await UniTask.Yield();
        var blocks = Map.FlowFieldBlocks;
        for (int i = 0; i < Map.MapSize.x; i++)
        {
            for (int j = 0; j < Map.MapSize.y; j++)
            {
                var block = Map.FlowFieldBlocks[i, j];
                if (block.BlockType == BlockType.Obstacle)
                    continue;

                var neighbors = Utils.GetNeighbors(new(i, j));
                if (block.Index == 1452)
                {
                    neighbors.ForEach(n =>
                    {
                        if (Utils.CheckValidPosition(n, Map.MapSize))
                            Debug.Log($"index {blocks[n.x, n.y].Index} HeatValue {blocks[n.x, n.y].HeatValue}");
                    });
                }

                var target = neighbors
                    .Where(neighbor => Utils.CheckValidPosition(neighbor, Map.MapSize) && blocks[neighbor.x, neighbor.y].BlockType != BlockType.Obstacle)
                    .OrderBy(neighbor => blocks[neighbor.x, neighbor.y].HeatValue)
                    .First();

                if (block.Index == 1452)
                {
                    Debug.Log($"{Map.FlowFieldBlocks[target.x, target.y].Index}");
                }
                block.SetDirection(target);
                if (DemoMode)
                    await UniTask.Delay(250);

            }
        }
    }

    private void CreateFlowFieldMap()
    {
        CreateFlowFieldMapAsync().Forget();
    }

    public async UniTask CreateFlowFieldMapAsync()
    {
        await MarkBlocksDistanceAsync();
        await MarkBlocksDirectionAsync();
    }
}
