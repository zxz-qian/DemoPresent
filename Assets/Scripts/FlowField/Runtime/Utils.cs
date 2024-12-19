using System.Collections.Generic;
using UnityEngine;
public static class Utils
{

    public static bool CheckValidPosition(Vector2Int position, Vector2Int mapSize)
    {
        return position.x >= 0 && position.x < mapSize.x
            && position.y >= 0 && position.y < mapSize.y;
    }

    /// <summary>
    ///        up
    /// left center right
    ///       down
    /// 
    /// left => up => right => down
    /// </summary>
    /// <param name="target"></param>
    /// <returns></returns> <summary>
    /// 
    /// </summary>
    /// <param name="target"></param>
    /// <returns></returns>
    public static List<Vector2Int> GetAdjacents(Vector2Int target)
    {
        var row = target.x;
        var column = target.y;

        var left = column - 1;
        var up = row + 1;
        var right = column + 1;
        var down = row - 1;

        return new()
        {
            new (row,left),
            new (up, column),
            new (row, right),
            new (down, column)
        };
    }

    /// <summary>
    /// 1   2    3
    /// 0 center 4
    /// 7   6    5
    /// </summary>
    /// <param name="target"></param>
    /// <returns></returns>
    public static List<Vector2Int> GetNeighbors(Vector2Int target)
    {
        var row = target.x;
        var column = target.y;
        var left = column - 1;
        var up = row + 1;
        var right = column + 1;
        var down = row - 1;
        return new()
        {
            new (row,left),
            new (up,left),
            new (up, column),
            new (up, right),
            new (row, right),
            new (down, right),
            new (down, column),
            new (down, left),
        };
    }
}