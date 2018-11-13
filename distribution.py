import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

def toLab(rgb):
    lab = np.uint8(np.round(rgb))
    size = lab.shape
    flag = len(size)<3
    if flag:
        lab = np.reshape(lab,(1,size[0],3))
    lab = cv2.cvtColor(lab,cv2.COLOR_RGB2LAB)
    lab = np.float64(lab)
    lab[:,:,0] *= 100/255
    lab[:,:,1:] -= 128
    if flag:
        lab = np.reshape(lab,(size[0],3))
    return lab

def around(div):
    s = int(180/6)
    inc = np.arange(s)/s*255
    dec = inc[::-1]
    ry = np.zeros((s,3))
    ry[:,0] = 255
    ry[:,1] = inc
    yg = np.zeros((s,3))
    yg[:,0] = dec
    yg[:,1] = 255
    gc = np.zeros((s,3))
    gc[:,1] = 255
    gc[:,2] = inc
    cb = np.zeros((s,3))
    cb[:,1] = dec
    cb[:,2] = 255
    bm = np.zeros((s,3))
    bm[:,0] = inc
    bm[:,2] = 255
    mr = np.zeros((s,3))
    mr[:,0] = 255
    mr[:,2] = dec
    around = np.vstack([ry,yg,gc,cb,bm,mr])
    lab = toLab(around)
    sign = -lab[:,2]
    sign[sign>=0] = 1
    sign[sign<0] = -1
    angle = sign*np.arccos(-lab[:,1]/np.linalg.norm(lab[:,1:],axis=1))*180/np.pi+180

    sort = np.argsort(angle)
    around = around[sort,:]
    angle = angle[sort]

    around = np.vstack((around[-1,:],around,around[0,:]))
    angle = np.hstack((angle[-1]-360,angle,angle[0]+360))

    dd = np.diff(angle)
    out = np.where(dd<=0)[0]+1
    around = np.delete(around,out,0)
    angle = np.delete(angle,out)

    p = 0
    color = np.zeros((div,3))
    for i in range(div):
        a = i*360/div
        while a>=angle[p+1]:
            p += 1
        d = (a-angle[p])/(angle[p+1]-angle[p])
        color[i,:] = (1-d)*around[p,:]+d*around[p+1,:]
    color /= 255
    color[color>1] = 1
    return color

def plot_dist_H(dist,N,div,color,name):
    plt.figure(2,figsize=(5,5))
    plt.tick_params(labelbottom="off",bottom="off")
    plt.tick_params(labelleft="off",left="off")
    plt.xticks([])
    ax = plt.gca()
    ax.spines["right"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    plt.scatter(0,0,s=100+100000*N**2,color=[0,0,0])
    z = np.arange(div)*np.pi*2/div

    x1 = np.cos(z)
    x2 = x1*0.8*(1-0.8*dist)
    y1 = np.sin(z)
    y2 = y1*0.8*(1-0.8*dist)
    for i in range(div):
        plt.plot([x2[i],x1[i]],[y2[i],y1[i]],linewidth=1,color=color[i,:])

    plt.plot(x2,y2,linewidth=1,color=[0.7,0.7,0.7])
    plt.plot(x1,y1,linewidth=1,color=[0,0,0])
    plt.plot(x1*0.8,y1*0.8,linewidth=1,color=[0,0,0])
    plt.savefig(name+'_H.jpg')

def H_dist(H,div):
    q = int(15/360*div)
    dist = np.zeros(div)
    for h in H:
        dist[int(h*div/360)-1] += 1
    dist = dist/np.sum(dist)
    idx = np.where(dist>0)[0]
    ddd = np.zeros(div+2*q)
    p = (np.arange(q)/q)**(5)
    p = np.hstack((p,[1],p[::-1]))
    dist = np.hstack((dist[-q:],dist,dist[:q]))
    for i in idx:
        ddd[i:i+2*q+1] += dist[i+q]*p
    ddd[q+1:2*q+1] += ddd[-q:]
    ddd[-2*q:-q] += ddd[:q]
    ddd = ddd[q:-q]
    ddd = ddd/np.max(ddd)
    return ddd

def VC_dist(pixel,d,V,C,L,name):
    vdiv = int(np.round(100/d)+1)
    cdiv = int(np.round(130/d)+1)

    cv = np.zeros((cdiv,vdiv))
    cc = np.zeros((cdiv,vdiv,3))
    point_c = (np.arange(cdiv*vdiv)/vdiv*d)
    point_v = (np.arange(cdiv*vdiv)%vdiv*d)
    for i in range(L):
        cv[C[i],V[i]] += 1
        cc[C[i],V[i],:] += pixel[i,:]
    idx = cv>0
    cc[idx] /= np.reshape(cv,(cdiv,vdiv,1))[idx]
    q = int(10*(1/d))
    xx = np.reshape((np.arange((2*q+1)**2)/(2*q+1)).astype(int),(2*q+1,2*q+1))-q
    yy = np.reshape((np.arange((2*q+1)**2)%(2*q+1)).astype(int),(2*q+1,2*q+1))-q
    p = (1-np.sqrt(xx**2+yy**2)/q)
    p[p<0] = 0
    p = p**(1/2)
    pp = np.transpose(np.tile(p,(3,1,1)),(1,2,0))
    dist = np.zeros((cdiv+2*q,vdiv+2*q))
    point = np.zeros((cdiv+2*q,vdiv+2*q))
    color = np.zeros((cdiv+2*q,vdiv+2*q,3))
    c,v = np.where(cv>0)
    for ci,vi in zip(c,v):
        c_ = ci+2*q+1
        v_ = vi+2*q+1
        dist[ci:c_,vi:v_] += p*cv[ci,vi]
        point[ci:c_,vi:v_] += p
        color[ci:c_,vi:v_] += pp*cc[ci,vi,:]
    dist = dist[q:-q,q:-q]
    dist /= np.max(dist)
    dist = (dist**(1/2))*255
    point = point[q:-q,q:-q]
    color = color[q:-q,q:-q,:]
    idx = point>0
    color[idx] /= np.reshape(point,(cdiv,vdiv,1))[idx]

    hsv = cv2.cvtColor(np.uint8(color*255),cv2.COLOR_RGB2HSV).astype(float)
    hsv[:,:,2] = dist
    hsv[:,:,1] *= (hsv[:,:,2]/255)
    hsv[:,:,2] = 255-hsv[:,:,2]+hsv[:,:,1]
    color = cv2.cvtColor(np.uint8(hsv),cv2.COLOR_HSV2RGB)/255
    color = np.transpose(color,(1,0,2))
    color = color[::-1,:,:]

    plt.figure(3,figsize=(6.65,5.15))
    plt.tick_params(labelbottom="off",bottom="off")
    plt.tick_params(labelleft="off",left="off")
    plt.imshow(np.uint8(color*255))
    plt.savefig(name+'_VC.jpg')

def color_dist(path,div_h,div_v):
    color = around(div_h)

    image = cv2.imread(path)
    while image.shape[0]*image.shape[1]>1200**2:
        image = cv2.resize(image, (int(image.shape[1]/2),int(image.shape[0]/2)))
        cv2.imwrite(path, image)
    image = np.array(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.figure(1)
    plt.imshow(image)
    image = np.reshape(image,(image.shape[0]*image.shape[1],3))
    L = image.shape[0]
    lab = toLab(image)
    V = np.round(lab[:,0]/div_v).astype(int)
    C = np.round(np.linalg.norm(lab[:,1:],axis=1)/div_v).astype(int)
    idx = np.linalg.norm(lab[:,1:],axis=1)>15
    N = 1-np.sum(idx)/lab.shape[0]
    lab_ = lab[idx,:]
    sign = -lab_[:,2]
    sign[sign>=0] = 1
    sign[sign<0] = -1
    xx = -lab_[:,1]/np.linalg.norm(lab_[:,1:],axis=1)
    xx[xx>1] = 1
    xx[xx<-1] = -1
    H = sign*np.arccos(xx)*180/np.pi+180
    H = np.ceil(H)

    name = path.split('.')[0]
    dist = H_dist(H,div_h)
    plot_dist_H(dist,N,div_h,color,name)
    VC_dist(np.uint8(image)/255,div_v,V,C,L,name)
    plt.show()

if __name__=='__main__':
    color_dist(sys.argv[1],1800,0.25)
