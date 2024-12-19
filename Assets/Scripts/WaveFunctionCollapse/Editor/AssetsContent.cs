using System;
using System.Collections.Generic;
using UnityEngine;

namespace WaveFunctionCollapse
{
    [CreateAssetMenu(fileName = "AssetsContent", menuName = "WaveFunctionCollapse/AssetsContent", order = 0)]
    public class AssetsContent : ScriptableObject
    {
        public List<AssetsContentData> assetsContentDatas;
    }

    [Serializable]
    public class AssetsContentData
    {
        public string name;

        public GameObject orignialAssetRef;

        public int index;

        public List<int> adjacents;

        public List<Vector3> adjacentsOffset;
    }
}