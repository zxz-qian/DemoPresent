using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace DrawSomething
{
    public class MeshDrawer : MonoBehaviour
    {
        void Start()
        {
            var cb = new CommandBuffer();
            cb.name = "OffscreenRender";
        }

        void Update()
        {

        }
    }

}
