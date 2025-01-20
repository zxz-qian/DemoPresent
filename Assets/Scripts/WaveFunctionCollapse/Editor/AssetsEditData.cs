using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace WaveFunctionCollapse
{
    [CreateAssetMenu(fileName = "AssetsEditData", menuName = "WaveFunctionCollapse/AssetsEditData", order = 0)]
    public class AssetsEditData : ScriptableObject
    {
        public const string SAVE_FILE_PATH = "Assets/Resources/WaveFunctionCollapse/AssetsEditData.asset";
        [SerializeField]
        private int EditObjectIndex;

        [SerializeField]
        private int AdjacentEnumIndex;
        [SerializeField]
        private int AdjacentObjectIndex;

    }
}