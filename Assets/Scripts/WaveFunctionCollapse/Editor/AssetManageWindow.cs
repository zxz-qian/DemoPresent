using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.Linq;
using System;
using Common;

namespace WaveFunctionCollapse
{
    public class AssetManageWindow : EditorWindow
    {
        private AssetsContent content;

        private GameObject tempSelectedObject = null;
        private AssetsContentData tempSelectedContentData = null;

        private GameObject tempAdjacentObject = null;

        private GameObject rootObject = null;

        private int curEditAdjacentEnumIdx;
        private GUIContent AdjecentDirLabel = new GUIContent("Direction");
        private List<GUIContent> RotateLables = new()
        {
            new GUIContent("X"),
            new GUIContent("Y"),
            new GUIContent("Z"),
        };
        private string SelectedObjectLabel = "Cur Editing Object";
        private GUIContent[] adjacentContent = new GUIContent[]
        {
            new GUIContent("Up"),
            new GUIContent("Down"),
            new GUIContent("Forward"),
            new GUIContent("Backward"),
            new GUIContent("Left"),
            new GUIContent("Right"),
        };

        private int[] adjacentOptions = new int[]
        {
            0,1,2,3,4,5
        };

        private int curSelectedObjectIdx;
        private string[] objectContentNames;
        private int[] objectContentOptions;

        private int curAdjacentObjectIdx;
        private int curEditSequence;

        private SerializedObject lastEditData;

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
            objectContentNames = new string[content.assetsContentDatas.Count];
            objectContentOptions = new int[content.assetsContentDatas.Count];
            for (int i = 0; i < content.assetsContentDatas.Count; i++)
            {
                objectContentNames[i] = content.assetsContentDatas[i].orignialAssetRef.name;
                objectContentOptions[i] = i;
            }

            rootObject = GameObject.Find("Root");

            var editData = AssetDatabase.LoadAssetAtPath<AssetsEditData>(AssetsEditData.SAVE_FILE_PATH);
            lastEditData = new SerializedObject(editData);

            curSelectedObjectIdx = lastEditData.FindProperty("EditObjectIndex").intValue;
            curEditAdjacentEnumIdx = lastEditData.FindProperty("AdjacentEnumIndex").intValue;

            RefreshSelectionData();
        }

        private void OnDestroy()
        {
            if (tempAdjacentObject != null)
                GameObject.DestroyImmediate(tempAdjacentObject);
        }

        private void OnGUI()
        {
            if (content == null)
                return;

            EditorGUI.BeginChangeCheck();
            {
                GUILayout.BeginHorizontal();
                {
                    curSelectedObjectIdx = EditorGUILayout.IntPopup(SelectedObjectLabel, curSelectedObjectIdx, objectContentNames, objectContentOptions);
                    if (GUILayout.Button("Focus"))
                    {
                        FocusObject(tempSelectedObject);
                    }
                }
                GUILayout.EndHorizontal();
            }
            var isChanged = EditorGUI.EndChangeCheck();
            if (isChanged)
            {
                lastEditData.FindProperty("EditObjectIndex").intValue = curSelectedObjectIdx;
                lastEditData.ApplyModifiedProperties();
                RefreshSelectionData();
                CreateAdjacentObject(curAdjacentObjectIdx);
            }
            EditorGUI.BeginChangeCheck();
            {
                curEditAdjacentEnumIdx = EditorGUILayout.IntPopup(AdjecentDirLabel, curEditAdjacentEnumIdx, adjacentContent, adjacentOptions);
            }

            isChanged = EditorGUI.EndChangeCheck();
            if (isChanged)
            {
                lastEditData.FindProperty("AdjacentEnumIndex").intValue = curEditAdjacentEnumIdx;
                lastEditData.ApplyModifiedProperties();

                CreateAdjacentObject(Mathf.Max(0, tempSelectedContentData.adjacentDataLists[curEditAdjacentEnumIdx].GetDataByIndex(curEditSequence).index));
            }

            DrawAdjanceIndexSelection();

            if (tempSelectedContentData == null)
                return;

            DrawChangeAdjanceObject();

            DrawObjectTransformingArea();

            DrawHandleAdjacentData();
        }

        private void DrawAdjanceIndexSelection()
        {
            // curEditSequence = EditorGUILayout.IntField("Sequence", curEditSequence);
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            {
                EditorGUILayout.LabelField("Sequence", EditorStyles.boldLabel);
                if (GUILayout.Button("<"))
                {
                    curEditSequence--;
                    curEditSequence %= tempSelectedContentData.adjacentDataLists[curEditAdjacentEnumIdx].datas.Count;
                }
                GUI.enabled = false;
                curEditSequence = EditorGUILayout.IntField("Sequence", curEditSequence);
                GUI.enabled = true;
                if (GUILayout.Button(">"))
                {
                    curEditSequence++;
                    curEditSequence %= tempSelectedContentData.adjacentDataLists[curEditAdjacentEnumIdx].datas.Count;

                }
            }
            EditorGUILayout.EndVertical();
        }

        private void DrawHandleAdjacentData()
        {
            EditorGUILayout.BeginHorizontal(EditorStyles.helpBox);
            {
                if (GUILayout.Button("Confirm"))
                {
                    SetAdjacentData();
                }

                if (GUILayout.Button("Reset"))
                {
                    SetAdjacentData(true);
                    CreateAdjacentObject(0);
                }
            }
            EditorGUILayout.EndHorizontal();
        }

        private void DrawChangeAdjanceObject()
        {
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            {
                EditorGUILayout.LabelField($"Change Adjance Object ({tempAdjacentObject.name.Replace("(Clone)", "")})", EditorStyles.boldLabel);
                EditorGUILayout.BeginHorizontal(EditorStyles.helpBox);
                {
                    if (GUILayout.Button("< Previous"))
                    {
                        curAdjacentObjectIdx--;
                        curAdjacentObjectIdx %= content.assetsContentDatas.Count;
                        CreateAdjacentObject(curAdjacentObjectIdx);
                    }

                    if (tempAdjacentObject == null)
                        CreateAdjacentObject(curAdjacentObjectIdx);

                    if (GUILayout.Button("Next >"))
                    {
                        curAdjacentObjectIdx++;
                        curAdjacentObjectIdx %= content.assetsContentDatas.Count;
                        CreateAdjacentObject(curAdjacentObjectIdx);
                    }
                }
                EditorGUILayout.EndHorizontal();
            }
            EditorGUILayout.EndVertical();
        }

        private void DrawObjectTransformingArea()
        {
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            {
                EditorGUILayout.BeginHorizontal();
                {
                    EditorGUILayout.LabelField($"Adjacent Object Rotation:", EditorStyles.boldLabel);
                    // GUILayout.FlexibleSpace();

                    if (GUILayout.Button("Reset"))
                    {
                        tempAdjacentObject.transform.rotation = Quaternion.identity;
                    }
                }
                EditorGUILayout.EndHorizontal();

                for (int i = 0; i < RotateLables.Count; i++)
                {
                    EditorGUILayout.BeginHorizontal(EditorStyles.helpBox);
                    {
                        GUILayout.FlexibleSpace();
                        var eulerAngle = Vector3.zero;
                        eulerAngle[i] = 90;
                        if (GUILayout.Button("-90°"))
                        {
                            var rotation = tempAdjacentObject.transform.rotation;
                            rotation *= Quaternion.Euler(eulerAngle * -1);
                            tempAdjacentObject.transform.rotation = rotation;
                        }
                        GUILayout.FlexibleSpace();

                        EditorGUILayout.LabelField(RotateLables[i], GUILayout.Width(20));
                        GUILayout.FlexibleSpace();

                        if (GUILayout.Button("+90°"))
                        {
                            var rotation = tempAdjacentObject.transform.rotation;
                            rotation *= Quaternion.Euler(eulerAngle);
                            tempAdjacentObject.transform.rotation = rotation;

                        }
                        GUILayout.FlexibleSpace();
                    }
                    EditorGUILayout.EndHorizontal();
                }
            }
            EditorGUILayout.EndVertical();
        }

        private void RefreshSelectionData()
        {
            tempSelectedContentData = content.assetsContentDatas[curSelectedObjectIdx];
            curAdjacentObjectIdx = tempSelectedContentData.adjacentDataLists[curEditAdjacentEnumIdx].GetDataByIndex(curEditSequence).index;
            curAdjacentObjectIdx = Mathf.Max(0, curAdjacentObjectIdx);
            // var name = tempSelectedContentData.orignialAssetRef.name;
            if (rootObject.transform.childCount > curSelectedObjectIdx)
            {
                tempSelectedObject = rootObject.transform.GetChild(curSelectedObjectIdx).gameObject;
                FocusObject(tempSelectedObject);
            }
        }

        private void FocusObject(GameObject gameObject)
        {
            Selection.activeGameObject = gameObject;
            var position = gameObject.transform.position;
            var size = Vector3.one * 2;
            var bounds = new Bounds(position, size);
            if (tempSelectedObject.TryGetComponent<MeshRenderer>(out var comp))
            {
                bounds = comp.bounds;
            }
            SceneView.lastActiveSceneView.Frame(bounds, false);
        }

        private void CreateAdjacentObject(int targetObjectIdx)
        {
            if (tempAdjacentObject != null)
                GameObject.DestroyImmediate(tempAdjacentObject);

            tempAdjacentObject = GameObject.Instantiate(content.assetsContentDatas[targetObjectIdx].orignialAssetRef);
            tempAdjacentObject.transform.SetParent(tempSelectedObject.transform);
            Selection.activeGameObject = tempAdjacentObject;
            var dataLists = tempSelectedContentData.adjacentDataLists[curEditAdjacentEnumIdx].datas;
            var sequence = dataLists.FindIndex(data => data.index == targetObjectIdx);

            var position = sequence < 0 ? Vector3.zero : dataLists[sequence].position;
            var rotation = sequence < 0 ? Quaternion.identity : dataLists[sequence].rotation;
            tempAdjacentObject.transform.SetPositionAndRotation(
                position + tempSelectedObject.transform.position,
                rotation
            );
        }

        private void ConfirmSettingData(AssetsContentData data)
        {
            content.assetsContentDatas[data.index] = data;
            AssetDatabase.SaveAssets();
        }


        private void SetAdjacentData(bool isReset = false)
        {
            var adjacentDataList = tempSelectedContentData.adjacentDataLists[curEditAdjacentEnumIdx];
            if (isReset)
            {
                if (adjacentDataList.datas.IsValidIndex(curEditSequence))
                {
                    adjacentDataList.GetDataByIndex(curEditSequence).index = -1;
                    adjacentDataList.GetDataByIndex(curEditSequence).position = Vector3.zero;
                    adjacentDataList.GetDataByIndex(curEditSequence).rotation = Quaternion.identity;
                }
            }
            else
            {
                if (adjacentDataList.datas.IsValidIndex(curEditSequence))
                {
                    adjacentDataList.GetDataByIndex(curEditSequence).index = curAdjacentObjectIdx;
                    adjacentDataList.GetDataByIndex(curEditSequence).position = tempAdjacentObject.transform.position - tempSelectedObject.transform.position;
                    adjacentDataList.GetDataByIndex(curEditSequence).rotation = tempAdjacentObject.transform.rotation;
                }
                else
                {
                    var data = new AdjacentData()
                    {
                        index = curAdjacentObjectIdx,
                        position = tempAdjacentObject.transform.position - tempSelectedObject.transform.position,
                        rotation = tempAdjacentObject.transform.rotation
                    };
                    adjacentDataList.datas.Add(data);
                }
            }
            ConfirmSettingData(tempSelectedContentData);

        }
    }
}