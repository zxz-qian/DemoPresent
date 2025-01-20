using System;
using System.Collections.Generic;
using Common;
using UnityEngine;

namespace WaveFunctionCollapse
{
    [CreateAssetMenu(fileName = "AssetsContent", menuName = "WaveFunctionCollapse/AssetsContent", order = 0)]
    public class AssetsContent : ScriptableObject
    {
        public List<AssetsContentData> assetsContentDatas = new();
    }

    [Serializable]
    public class AssetsContentData
    {
        public string name;

        public GameObject orignialAssetRef;

        public int index;

        // public List<int> adjacents;

        // public List<Vector3> adjacentsOffset;
        // public List<Quaternion> adjacentsRotation;

        public AdjacentDataList[] adjacentDataLists = new AdjacentDataList[6]
        {
            new (),
            new (),
            new (),
            new (),
            new (),
            new (),
        };
    }

    [Serializable]
    public class AdjacentDataList
    {
        public List<AdjacentData> datas = new();

        public AdjacentData GetDataByIndex(int idx)
        {
            if (datas.IsValidIndex(idx))
            {
                return datas[idx];
            }
            else
            {
                return AdjacentData.Default;
            }
        }
    }

    [Serializable]
    public class AdjacentData
    {
        public static AdjacentData Default = new AdjacentData()
        {
            index = -1,
            position = Vector3.zero,
            rotation = Quaternion.identity
        };
        public int index;
        public Vector3 position;
        public Quaternion rotation;
    }
}