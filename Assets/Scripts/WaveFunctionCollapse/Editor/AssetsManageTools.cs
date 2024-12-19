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

            var assetsContent = Resources.Load<AssetsContent>("WaveFunctionCollapse/AssetsContent");
            assetsContent.assetsContentDatas = new();
            for (int i = 0; i < count; i++)
            {
                var objName = obj.transform.GetChild(i).name;
                var assetsContentData = new AssetsContentData
                {
                    index = i,
                    name = obj.transform.GetChild(i).name,
                    orignialAssetRef = AssetDatabase.LoadAssetAtPath<GameObject>($"{AssetDatapathPrefix}{objName}{AssetDatapathSuffix}"),
                    adjacents = new() { -1, -1, -1, -1, -1, -1 },
                    adjacentsOffset = new() { Vector3.zero, Vector3.zero, Vector3.zero, Vector3.zero, Vector3.zero, Vector3.zero }
                };
                assetsContent.assetsContentDatas.Add(assetsContentData);
            }
        }
    }
}