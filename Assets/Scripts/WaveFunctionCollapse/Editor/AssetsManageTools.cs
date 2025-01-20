using UnityEditor;
using UnityEngine;

namespace WaveFunctionCollapse
{
    public class AssetsManageTools
    {
        private const string AssetDatapathPrefix = "Assets/Models/WaveFunctionCollapse/CastleKit/";
        private const string AssetDatapathSuffix = ".fbx";

        [MenuItem("WaveFunctionCollapse/FlattenAssets")]
        public static void FlattenAssets()
        {
            var obj = Selection.activeGameObject;
            var count = obj.transform.childCount;
            var size = Mathf.FloorToInt(Mathf.Sqrt(count));
            var spaceSize = 3;

            for (int i = 0; i < count; i++)
            {
                obj.transform.GetChild(i).transform.position = new Vector3(i % size * spaceSize, 0, i / size * spaceSize);
            }
        }

        [MenuItem("WaveFunctionCollapse/RecordAssets")]
        public static void RecordAssets()
        {
            var obj = Selection.activeGameObject;
            var count = obj.transform.childCount;

            var asset = AssetDatabase.LoadAssetAtPath<AssetsContent>("Assets/Resources/WaveFunctionCollapse/AssetsContent.asset");

            var so = new SerializedObject(asset);
            var datas = so.FindProperty("assetsContentDatas");

            for (int i = 0; i < count; i++)
            {
                var objName = obj.transform.GetChild(i).name;
                datas.InsertArrayElementAtIndex(i);
                // datas.
                var data = datas.GetArrayElementAtIndex(i);
                data.FindPropertyRelative("name").stringValue = objName;
                data.FindPropertyRelative("orignialAssetRef").objectReferenceValue = AssetDatabase.LoadAssetAtPath<GameObject>($"{AssetDatapathPrefix}{objName}{AssetDatapathSuffix}");
                data.FindPropertyRelative("index").intValue = i;

                var adjacentDataLists = data.FindPropertyRelative("adjacentDataLists");
                adjacentDataLists.ClearArray();
                for (int j = 0; j < 6; j++)
                {
                    adjacentDataLists.InsertArrayElementAtIndex(j);
                }
            }
            so.ApplyModifiedProperties();
            AssetDatabase.SaveAssets();
        }
    }
}