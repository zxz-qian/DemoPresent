using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.Linq;
using System.Data.Common;
using System;

namespace WaveFunctionCollapse
{
    public class AssetManageWindow : EditorWindow
    {
        private AssetsContent content;

        private GameObject tempSelectedObject = null;
        private AssetsContentData tempSelectedContentData = null;
        private List<int> tempAdjacentIdx = new();
        private List<GameObject> tempAdjacentObjects = new();

        [MenuItem("WaveFunctionCollapse/AssetManageWindow")]
        private static void ShowWindow()
        {
            var window = GetWindow<AssetManageWindow>();
            window.titleContent = new GUIContent("WaveFunctionCollapse");
            window.Show();
        }

        private void Awake()
        {
            content = Resources.Load<AssetsContent>("WaveFunctionCollapse/AssetsContent");
        }

        private void OnGUI()
        {
            if (content == null)
                return;

            var data = GetAssetsContentData();
            if (data == null)
                return;

            for (int i = 0; i < tempAdjacentIdx.Count; i++)
            {
                EditorGUILayout.BeginHorizontal();
                // var adjacentIndex = data.adjacents[i];

                PlaceAdjacentObject(data);

                if (GUILayout.Button("< previous"))
                {

                }
                if (GUILayout.Button("Next >"))
                {

                }

                if (GUILayout.Button("Confirm"))
                {
                    // data.
                    ConfirmSettingData(data);
                }
                EditorGUILayout.EndHorizontal();

            }
        }

        private void PlaceAdjacentObject(AssetsContentData data)
        {
            for (int i = 0; i < tempAdjacentIdx.Count; i++)
            {
                // if (tempAdjacentIdx[i] ==)
            }
        }

        private AssetsContentData GetAssetsContentData()
        {
            var obj = Selection.activeGameObject;
            if (tempSelectedObject != obj)
            {
                tempSelectedContentData = content.assetsContentDatas.FirstOrDefault(data => data.name == obj.name);
                tempAdjacentIdx = tempSelectedContentData?.adjacents;
            }

            return tempSelectedContentData;
        }

        private void ConfirmSettingData(AssetsContentData data)
        {
            content.assetsContentDatas[data.index] = data;
        }

        private void GenerateTempObject(AssetsContentData data, Vector3 position)
        {
            // if (tempObjectIndex == data.index)
            //     return;
            // else
            // {
            //     if (tempObject != null)
            //         GameObject.DestroyImmediate(tempObject);
            // }

            // tempObjectIndex = data.index;
            // tempObject = GameObject.Instantiate<GameObject>(data.orignialAssetRef);
            // tempObject.transform.position = position;
        }
    }
}