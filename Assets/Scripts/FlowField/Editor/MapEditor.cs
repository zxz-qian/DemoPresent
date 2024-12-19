using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System.Linq;

public class MapEditor
{
    [MenuItem("FlowField/CreateMap")]
    private static void CreateMap()
    {
        var map = GameObject.FindObjectOfType<MapGenerator>();
        map.MakeMap();
    }

    [MenuItem("FlowField/ClearMap")]
    private static void ClearMap()
    {
        var map = GameObject.FindObjectOfType<MapGenerator>();
        for (int i = 0; i < map.transform.childCount; i++)
        {
            GameObject.DestroyImmediate(map.transform.GetChild(i).gameObject);
        }
    }

    [MenuItem("FlowField/MarkAsObstacle")]
    private static void MarkAsObstacle()
    {
        var objs = Selection.gameObjects;
        var map = GameObject.FindObjectOfType<MapGenerator>();

        objs.ToList().ForEach(obj =>
        {
            if (obj.TryGetComponent<FlowFieldBlock>(out var comp))
            {
                comp.Setup(map, 0);
                comp.SetBlockType(BlockType.Obstacle);
            }
            else if (obj.transform.parent.TryGetComponent<FlowFieldBlock>(out comp))
            {
                comp.Setup(map, 0);
                comp.SetBlockType(BlockType.Obstacle);
            }
        });
    }

    [MenuItem("FlowField/MarkAsNormal")]
    private static void MarkAsNormal()
    {
        var objs = Selection.gameObjects;
        var map = GameObject.FindObjectOfType<MapGenerator>();
        objs.ToList().ForEach(obj =>
        {
            if (obj.TryGetComponent<FlowFieldBlock>(out var comp))
            {
                comp.Setup(map, 0);
                comp.SetBlockType(BlockType.Normal);
            }
            else if (obj.transform.parent.TryGetComponent<FlowFieldBlock>(out comp))
            {
                comp.Setup(map, 0);
                comp.SetBlockType(BlockType.Normal);
            }
        });
    }


    [MenuItem("FlowField/MarkAsEnd")]
    private static void MarkAsEnd()
    {
        var objs = Selection.gameObjects;
        var map = GameObject.FindObjectOfType<MapGenerator>();
        objs.ToList().ForEach(obj =>
        {
            if (obj.TryGetComponent<FlowFieldBlock>(out var comp))
            {
                comp.Setup(map, 0);
                comp.SetBlockType(BlockType.End);
            }
            else if (obj.transform.parent.TryGetComponent<FlowFieldBlock>(out comp))
            {
                comp.Setup(map, 0);
                comp.SetBlockType(BlockType.End);
            }
        });
    }
}


