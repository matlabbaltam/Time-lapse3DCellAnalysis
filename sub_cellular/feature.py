import numpy as np
from scipy import ndimage
from scipy.ndimage import measurements


#######################################################################





from scipy import ndimage
from scipy.ndimage import measurements
import SimpleITK as sitk
from skimage import segmentation
from skimage import io
import os
import pandas as pd



def adj_list_to_matrix(adj_list):
    n = len(adj_list)
    adj_matrix = np.zeros((n,n))
    np.fill_diagonal(adj_matrix,0)
    for i in range(1,n+1):
        for j in adj_list[i]:
            adj_matrix[i-1,j-1] = 1
    return adj_matrix

def compute_cell_adjacent_table(seg_img):
    #seg_img[seg_img==17]=15
    Adjacent_table={}
    all_labels = np.unique(seg_img)
    print all_labels
    #all_labels = np.delete(all_labels,all_labels==0)
    for i in all_labels:
        if i != 0:
            index_list = []
            for j in all_labels:
                if j !=0:
                    draw_board = np.zeros(seg_img.shape)
                    if i!=j:
                        draw_board[seg_img==i]=1
                        draw_board[seg_img==j]=1
                        draw_board = ndimage.binary_dilation(draw_board).astype(draw_board.dtype) 
                        _,num = measurements.label(draw_board)
                        if num==1:
                            index_list.append(j)
                tmp_dict={}
                tmp_dict[i]=index_list
                Adjacent_table.update(tmp_dict)
    return Adjacent_table

def compute_contact_points(seg_img,Adj_matrix):
    n = len(Adj_matrix)
    [slices,x,y] = seg_img.shape
    wall = np.zeros([x,y])
    for i in range(n):
        for j in range(n):
            if Adj_matrix[i,j]==1:
                draw_board = np.zeros([slices,x,y])
                draw_contact = np.zeros([x,y])
                for k in range(slices-1,-1,-1):
                    #fig = plt.figure()
                    draw_board1 = np.zeros([x,y])
                    draw_board2 = np.zeros([x,y])
                    draw_board1[seg_img[k]==i+1]=1
                    draw_board1 = ndimage.binary_dilation(draw_board1).astype(draw_board1.dtype)
                    draw_board2[seg_img[k]==j+1]=1
                    #fig.add_subplot(3,1,1)
                    #plt.imshow(draw_board1,cmap='gray')
                    #fig.add_subplot(3,1,2)
                    #plt.imshow(draw_board2,cmap='gray')
                    draw_board2 = ndimage.binary_dilation(draw_board2).astype(draw_board2.dtype)
                    draw_board[k,:,:] = np.logical_and(draw_board1==1,draw_board2==1)
                    draw_board[k,:,:] = draw_board[k,:,:]*1
                    #filename =  "adjacent_cell_bound/{}{}slice{}.png".format(i+1,j+1,k+1)
                    #io.imsave(filename,draw_board[k,:,:])
                    #print "cell {} and cell {} in slice {}".format(i+1,j+1,k+1)
                    #fig.add_subplot(3,1,3)
                    #plt.imshow(draw_board,cmap='gray')
                    #plt.show()
                    #_,num = measurements.label(draw_board)
                    #if num==1:
                    #    wall = segmentation.find_boundaries(draw_board,connectivity=1,mode="thick")
                    #    break
                bound_len = np.zeros(slices)
                for kk in range(slices):
                    bound_len[kk] =draw_board[kk,:,:].sum()
                print bound_len
                for kk in range(slices-1,0,-1):
                    if kk < slices-1:
                        if bound_len[kk]>bound_len[kk+1] and bound_len[kk]>bound_len[kk-1]:
                            draw_contact_temp = draw_board[kk,:,:]
                            draw_contact = np.zeros(draw_board.shape)
                            for jj in range(len(draw_contact)):
                                draw_contact[jj] = draw_contact_temp
                            filename =  "data5_contact/{}{}slice{}.nii.gz".format(i+1,j+1,kk+1)
                            draw_contact_img = sitk.GetImageFromArray((draw_contact*255).astype('uint8'))
                            sitk.WriteImage(draw_contact_img,filename)
                            break
    return wall




def compute_conjunction_points(seg_img,Adj_list):
#     print "compute conjunction points"
    #seg_img[seg_img==17]=15
    [slices,x,y] = seg_img.shape
    n = len(Adj_list)
    plane = seg_img[int(len(seg_img)/2+1)]
    final = np.zeros((x,y))
    final_dict = {}
    for i in Adj_list.keys():
        A = i
        A_neighbors = Adj_list[A]
        for j in range(len(A_neighbors)):
            B = A_neighbors[j]
            B_neighbors = Adj_list[B]
            for k in range(len(B_neighbors)):
                C = B_neighbors[k]
                draw_board1 = np.zeros((x,y))
                draw_board2 = np.zeros((x,y))
                draw_board3 = np.zeros((x,y))
                draw_board = np.zeros((x,y))
                if C in A_neighbors:
                    draw_board1[plane==A]=1
                    draw_board2[plane==B]=1
                    draw_board3[plane==C]=1
                    draw_board1 = ndimage.binary_dilation(draw_board1,iterations=1).astype(draw_board1.dtype)
                    draw_board2 = ndimage.binary_dilation(draw_board2,iterations=1).astype(draw_board2.dtype)
                    draw_board3 = ndimage.binary_dilation(draw_board3,iterations=1).astype(draw_board3.dtype)
                    #print "cell {} {} {}".format(A,B,C)
                    #plt.imshow(draw_board1)
                    #plt.show()
                    draw_board =  np.logical_and(draw_board1==1,draw_board2==1)
                    draw_board = np.logical_and(draw_board==1,draw_board3==1)
                    draw_board = draw_board*1
                    point = np.nonzero(draw_board)
                    if len(point[0])>0:
                        final_dict["{} {} {}".format(A,B,C)]=point
                    #final = np.logical_or(draw_board,final)
    #final = final*1
    #points = np.nonzero(final)
    return final_dict

def cell_volumn(seg_img):
    print('Compute each cell volumn')
    all_labels = np.unique(seg_img)
    cell_volumn = []
    unique,counts = np.unique(seg_img,return_counts=True)
    result = dict(zip(unique,counts))
    result.pop(0)
    return result


def cell_center(seg_img):
    results = {}
    for label in np.unique(seg_img):
        if label != 0:
            all_points_z,all_points_x,all_points_y = np.where(seg_img==label)
            avg_z = np.round(np.mean(all_points_z))
            avg_x = np.round(np.mean(all_points_x))
            avg_y = np.round(np.mean(all_points_z))
            results[label]=[avg_z,avg_x,avg_y]
    return results
            



def seg_img_coord(seg_img):
    coords = {}
    for label in np.unique(seg_img):
        if label !=0:
            coordinate = np.argwhere(seg_img==label)
            coords[label] = coordinate
    return coords 


def relabel(seg_img):
    new_label = 0
    for old_label in np.unique(seg_img):
        seg_img[seg_img==old_label] = new_label
        new_label = new_label+1
    return seg_img


def compute_segments(seg_img, Adj_list):
    [slices, x, y] = seg_img.shape
    final_dict = {}
    plane = seg_img[int(len(seg_img) / 2 + 1)]
    for i in Adj_list.keys():
        A = i
        A_neighbors = Adj_list[A]
        for j in range(len(A_neighbors)):
            B = A_neighbors[j]
            #draw_board = np.zeros([slices, x, y])
            draw_contact = np.zeros([x, y])
            #for k in range(slices - 1, -1, -1):
            # fig = plt.figure()
            draw_board1 = np.zeros([x, y])
            draw_board2 = np.zeros([x, y])
            draw_board1[plane== A] = 1
            draw_board1 = ndimage.binary_dilation(draw_board1).astype(draw_board1.dtype)
            draw_board2[plane == B] = 1
            draw_board2 = ndimage.binary_dilation(draw_board2).astype(draw_board2.dtype)
            draw_contact = np.logical_and(draw_board1 == 1, draw_board2 == 1)
            draw_contact= draw_contact * 1
            point = np.nonzero(draw_contact)
            if len(point[0])>0:
                point_list = []
                for ii in range(len(point[0])):
                    point_list.append([point[0][ii],point[1][ii]])
                point_list = np.asarray(point_list)
                final_dict["{} {}".format(A,B)] = point_list
    return final_dict

def main(seg_dir):
    img = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img)
    masks = np.asarray(img,dtype='int')
    num_cells = len(np.unique(masks))-1
    coordinates = seg_img_coord(masks)
    #for label in np.unique(seg):
    adj_table = compute_cell_adjacent_table(masks)
    adj_table_done = tuple([(k, v) for k, v in adj_table.items()])
    #df = pd.DataFrame(adj_table)
    #df.to_csv("result/adj_table.csv")
    center = cell_center(masks)
    points = compute_conjunction_points(masks,adj_table)
    points_done = tuple([(k, [x for xs in v for x in xs]) for k, v in points.items()] )
    #np.savetxt("result/points.csv", points, delimiter=",")

    cell_vol = cell_volumn(masks)
    cell_vol_done = tuple([(k, v) for k, v in cell_vol.items()] )
    segments = compute_segments(masks, adj_table)
    #df1 = pd.DataFrame(cell_volumn(masks))
    #df1.to_csv("result/cell_volumn.csv")
	#output_files.append('source/result/'+file+'seg.tif')
    tfu = adj_table.values()
    b = np.full([len(tfu),len(max(tfu,key = lambda x: len(x)))], fill_value=np.nan)
	for i,j in enumerate(tfu):
        b[i][0:len(j)] = j

    with h5py.File('source/hdf/PlantCellSegmentation.h5', 'w') as hf:
	    grp0 = hf.create_group("Cell Labels")
	    grp1 = hf.create_group("Surface Coordinates")
	    grp2 = hf.create_group("Cell Center")
	    grp3 = hf.create_group("Cell Volume")
	    grp4 = hf.create_group("Adjacency Table")
	    grp5 = hf.create_group("Segmented Image")
	    grp2.create_dataset("Cell Center",  data=(np.array((center.values()))))
	    grp3.create_dataset("Cell Volume",  data=(np.array((cell_vol.values()))))
	    grp4.create_dataset("Adjacency Table", data=b)
	    grp0.create_dataset("Cell Labels", data=np.array((center.keys())))
	    grp5.create_dataset("Segmented Image", data=masks,compression='gzip',compression_opts=9)
    	for ix in range(num_cells):
                grp1.create_dataset("Cell Label: {}".format(ix), compression='gzip',compression_opts=9,  data=(np.array((coordinates.values())))[ix])

        #outtable_xml_adj_table = table_service.store_array(adj_table_done, name='adj_table')
        #outtable_xml_points = table_service.store_array(points_done, name='points')
        #outtable_xml_cell_vol = table_service.store_array(cell_vol_done, name='cell_vol')
    #print(output_files)
    return adj_table, points, cell_vol, coordinates, center,segments


if __name__== "__main__":
    seg_dir = "result/example.tif"
    main(seg_dir)