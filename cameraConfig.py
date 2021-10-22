# by Facundo Sosa-Rey, 2021. MIT license

camConfig={
    "30x30x200_pan":{
        "position":         [100, -20., 100.],
        "focal_point":      [50., 4., 60.],
        "view_angle" :      30.0,
        "view_up" :         [-0.13374911312082394, -0.940784157442076, -0.3115065711111029],
        "clipping_range":   [44.066053974203356, 465.8068559510238]
        },
    "231x231x81_pan":{
        "position":         [251.44098373135466, -78.41124167838605, 204.92811750355918],
        "focal_point":      [102.32180324137106, 24.951687619486705, 59.77226644750877],
        "view_angle" :      30.0,
        "view_up" :         [-0.29238260848920156, -0.8949795085879614, -0.33693336056363055],
        "clipping_range":   [0.5935350822428739, 593.5350822428738]
        },
    "231x231x81_static":{
        "position":         [326.8852884451577, -198.11248527495655, 277.82570748513285],
        "focal_point":      [51.5137425977142, 59.16074129470711, 111.89321440197402],
        "view_angle" :      30.0,
        "view_up" :         [-0.509102946402276, -0.7798370061273547, -0.36423678265497217],
        "clipping_range":   [185.0613654960054, 695.9853734027126]
        },
    "231x231x81_InclinedLargeFiber":{        
        "position":         [284.2678659674053, 307.1927056993693, 625.464543927443],
        "focal_point":      [129.91656032172583, 96.62353700139234, 4.29282386194221],
        "view_angle" :      30.0,
        "view_up" :         [-0.7776093682737375, -0.5109370496505462, 0.3664246193520768],
        "clipping_range":   [425.5807855547908, 902.5626448750932]
    },
    "231x231x81_InclinedLargeFiber_permuted132":{        
        "position":         [475.8866322320294, 449.3951835880689, 308.1047074046721],
        "focal_point":      [50.683937634910876, 144.67477063469778, 117.18968539519143],
        "view_angle" :      30.0,
        "view_up" :         [0.6339433387458596, -0.7362348083503988, -0.23679981045812531],
        "clipping_range":   [303.8139515159967, 930.3061508545932]
    },
    "manual_0":{
        "position":         [617.7277778989019, 425.1908124211711, 354.7099058534857],
        "focal_point":      [413.83264118283984, 241.5628856228669, 105.86415675761184],
        "view_angle" :      30.0,
        "view_up" :         [-0.4407765214750945, -0.5108141340776043, 0.7380955077378961],
        "clipping_range":   [1.0547172405897414, 1054.7172405897413]

    }

}


def shiftCamera(camConfigKey,scene,zHeigth):

    position=camConfig[camConfigKey]["position"].copy()
    position[2]+=zHeigth
    focal_point=camConfig[camConfigKey]["focal_point"].copy()
    focal_point[2]+=zHeigth
    scene.scene.camera.position = position
    scene.scene.camera.focal_point = focal_point
    scene.scene.camera.view_angle = camConfig[camConfigKey]["view_angle"]
    scene.scene.camera.view_up = camConfig[camConfigKey]["view_up"]
    # scene.scene.camera.clipping_range = camConfig[camConfigKey]["clipping_range"] 
    scene.scene.camera.compute_view_plane_normal()
    scene.scene.render()

def createCamViewFromOutline(rangeOutline,permuteVec,cameraConfigKey):
    if permuteVec=="123":
        camConfig[cameraConfigKey]={
            "position":     [rangeOutline[3]+(rangeOutline[3]-rangeOutline[2])*2.,rangeOutline[1]+(rangeOutline[1]-rangeOutline[0])*2.,rangeOutline[5]+(rangeOutline[5]-rangeOutline[4])*.5],
            "focal_point":  [rangeOutline[0]+(rangeOutline[1]-rangeOutline[0])*.5,rangeOutline[2]+(rangeOutline[3]-rangeOutline[2])*.5,rangeOutline[4]+(rangeOutline[5]-rangeOutline[4])*(0.)],
            "view_angle" :  30.0,
            "view_up":      [0., 0., 1.],
        }


    if permuteVec=="132":
        camConfig[cameraConfigKey]={
            "position":     [(rangeOutline[1]-rangeOutline[0])*6.,(rangeOutline[3]-rangeOutline[2])*2.4,(rangeOutline[5]-rangeOutline[4])*1.25],
            "focal_point":  [(rangeOutline[1]-rangeOutline[0])*0.5,(rangeOutline[3]-rangeOutline[2])*0.6,(rangeOutline[5]-rangeOutline[4])*0.5],
            "view_angle" :  26.0,
            "view_up":      [1., 0., 0.],
        }

    if permuteVec=="321":
        camConfig[cameraConfigKey]={
            "position":     [(rangeOutline[1]-rangeOutline[0])*2.4,(rangeOutline[3]-rangeOutline[2])*6.,(rangeOutline[5]-rangeOutline[4])*1.25],
            "focal_point":  [(rangeOutline[1]-rangeOutline[0])*0.66,(rangeOutline[3]-rangeOutline[2])*0.5,(rangeOutline[5]-rangeOutline[4])*0.5],
            "view_angle" :  26.0,
            "view_up":      [0., 1., 0.],
        }

    print("rangeOutline=",rangeOutline)
    print("camera config=",camConfig[cameraConfigKey])