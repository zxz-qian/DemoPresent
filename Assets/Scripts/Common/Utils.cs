

using System.Collections;
using System.Collections.Generic;

namespace Common
{
    public static class Utils
    {
        public static bool IsValidIndex<T>(this List<T> list, int idx)
        {
            return list != null && idx >= 0 && idx < list.Count;
        }
    }
}