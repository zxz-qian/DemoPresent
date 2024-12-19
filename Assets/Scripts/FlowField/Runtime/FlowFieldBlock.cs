using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using DG.Tweening;
using Unity.VisualScripting;

public enum ContentType
{
    Index = 0,
    HeatMap = 1,
    Direction = 2,
}

public class FlowFieldBlock : MonoBehaviour
{
    [SerializeField]
    private TextMeshPro textComp;
    [SerializeField]
    private MeshRenderer meshRenderer;
    [SerializeField]
    private Transform arrow;

    private int idx;
    public int Index => idx;

    private int heatValue = -1;
    public int HeatValue => heatValue;

    [SerializeField]
    private BlockType blockType = BlockType.Normal;
    public BlockType BlockType => blockType;
    [SerializeField]
    private Vector2 direction;
    [SerializeField]
    private int targetIndex;
    // private Vector2Int target;
    private Quaternion rotation;
    public Vector2 Direction => direction;

    private MapGenerator MapGenerator;

    public MeshRenderer MeshRenderer => meshRenderer;

    private ContentType contentType;

    private void Start()
    {
        // if (BlockType == BlockType.Obstacle)
        // {
        //     transform.AddComponent<BoxCollider>();
        // }
    }

    public void SetupText(string text)
    {
        textComp.text = text;
    }

    public void Setup(MapGenerator mapGenerator, int index)
    {
        MapGenerator = mapGenerator;
        idx = index;
        SetupText(idx.ToString());
    }

    public void SetBlockType(BlockType blockType)
    {
        this.blockType = blockType;
        if (MapGenerator == null)
            return;
        meshRenderer.material = MapGenerator.GetMaterial(blockType);
    }

    public void SetHeatValue(int value)
    {
        heatValue = value;
        // HightlightBlock();
    }

    public void SetDirection(Vector2Int target)
    {
        var column = idx % MapGenerator.MapSize.x;
        var row = idx / MapGenerator.MapSize.y;
        targetIndex = MapGenerator.FlowFieldBlocks[target.x, target.y].Index;
        direction = new Vector2Int(target.x - column, target.y - row);
        var sign = Mathf.Sign(Vector3.Dot(Vector3.Cross(Vector3.left, new Vector3Int(target.x - column, 0, target.y - row)), Vector3.up));
        rotation = Quaternion.Euler(0, Vector2.Angle(direction, Vector2.left) * sign, 0);
    }

    public void HightlightBlock()
    {
        var finalColor = meshRenderer.material.color;
        meshRenderer.material.color = Color.red;
        meshRenderer.material.DOColor(finalColor, 0.5f);
    }

    private void Update()
    {
        switch (contentType)
        {
            case ContentType.Index:
                SetupText(Index.ToString());
                arrow.gameObject.SetActive(false);
                textComp.color = Color.white;
                break;
            case ContentType.HeatMap:
                SetupText(heatValue.ToString());
                arrow.gameObject.SetActive(false);
                textComp.color = Color.red;
                break;
            case ContentType.Direction:
                arrow.gameObject.SetActive(true);
                arrow.transform.rotation = rotation;
                break;
            default:
                break;
        }
    }

    public void ShowContent(ContentType contentType)
    {
        this.contentType = contentType;
    }

}
