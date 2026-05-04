//
//  convolution.c
//
//
//  Created by Josep Lluis Lerida on 11/03/15.
//
// This program calculates the convolution for PPM images.
// The program accepts an PPM image file, a text definition of the kernel matrix and the PPM file for storing the convolution results.
// The program allows to define image partitions for processing large images (>500MB)
// The 2D image is represented by 1D vector for chanel R, G and B. The convolution is applied to each chanel separately.

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

// Estructura per emmagatzemar el contingut d'una imatge.
struct imagenppm{
    int altura;
    int ancho;
    char *comentario;
    int maxcolor;
    int P;
    int *R;
    int *G;
    int *B;
};
typedef struct imagenppm* ImagenData;

// Estructura per emmagatzemar el contingut d'un kernel.
struct structkernel{
    int kernelX;
    int kernelY;
    float *vkern;
};
typedef struct structkernel* kernelData;

//Functions Definition
ImagenData initimage(char* nombre, FILE **fp, int partitions, int halo);
ImagenData duplicateImageData(ImagenData src, int partitions, int halo);

int readImage(ImagenData Img, FILE **fp, int dim, int halosize, long int *position);
int duplicateImageChunk(ImagenData src, ImagenData dst, int dim);
int initfilestore(ImagenData img, FILE **fp, char* nombre, long *position);
int savingChunk(ImagenData img, FILE **fp, int dim, int offset);
int convolve2D(int* inbuf, int* outbuf, int sizeX, int sizeY, float* kernel, int ksizeX, int ksizeY);
void freeImagestructure(ImagenData *src);

//Open Image file and image struct initialization
ImagenData initimage(char* nombre, FILE **fp,int partitions, int halo){
    char c;
    char comentario[300];
    int i=0,chunk=0;
    ImagenData img=NULL;
    
    /*Se habre el fichero ppm*/

    if ((*fp=fopen(nombre,"r"))==NULL){
        perror("Error: ");
    }
    else{
        //Memory allocation
        img=(ImagenData) malloc(sizeof(struct imagenppm));

        //Reading the first line: Magical Number "P3"
        fscanf(*fp,"%c%d ",&c,&(img->P));
        
        //Reading the image comment
        while((c=fgetc(*fp))!= '\n'){comentario[i]=c;i++;}
        comentario[i]='\0';
        //Allocating information for the image comment
        img->comentario = calloc(strlen(comentario),sizeof(char));
        strcpy(img->comentario,comentario);
        //Reading image dimensions and color resolution
        fscanf(*fp,"%d %d %d",&img->ancho,&img->altura,&img->maxcolor);
        chunk = img->ancho*img->altura / partitions;
        //We need to read an extra row.
        chunk = chunk + img->ancho * halo;
        if ((img->R=calloc(chunk,sizeof(int))) == NULL) {return NULL;}
        if ((img->G=calloc(chunk,sizeof(int))) == NULL) {return NULL;}
        if ((img->B=calloc(chunk,sizeof(int))) == NULL) {return NULL;}
    }
    return img;
}

//Duplicate the Image struct for the resulting image
ImagenData duplicateImageData(ImagenData src, int partitions, int halo){
    char c;
    char comentario[300];
    unsigned int imageX, imageY;
    int i=0, chunk=0;
    //Struct memory allocation
    ImagenData dst=(ImagenData) malloc(sizeof(struct imagenppm));

    //Copying the magic number
    dst->P=src->P;
    //Copying the string comment
    dst->comentario = calloc(strlen(src->comentario),sizeof(char));
    strcpy(dst->comentario,src->comentario);
    //Copying image dimensions and color resolution
    dst->ancho=src->ancho;
    dst->altura=src->altura;
    dst->maxcolor=src->maxcolor;
    chunk = dst->ancho*dst->altura / partitions;
    //We need to read an extra row.
    chunk = chunk + src->ancho * halo;
    if ((dst->R=calloc(chunk,sizeof(int))) == NULL) {return NULL;}
    if ((dst->G=calloc(chunk,sizeof(int))) == NULL) {return NULL;}
    if ((dst->B=calloc(chunk,sizeof(int))) == NULL) {return NULL;}
    return dst;
}

//Read the corresponding chunk from the source Image
int readImage(ImagenData img, FILE **fp, int dim, int halosize, long *position){
    int i=0, k=0,haloposition=0;
    if (fseek(*fp,*position,SEEK_SET))
        perror("Error: ");
    haloposition = dim-(img->ancho*halosize*2);
    for(i=0;i<dim;i++) {
        // When start reading the halo store the position in the image file
        if (halosize != 0 && i == haloposition) *position=ftell(*fp);
        fscanf(*fp,"%d %d %d ",&img->R[i],&img->G[i],&img->B[i]);
        k++;
    }
//    printf ("Readed = %d pixels, posicio=%lu\n",k,*position);
    return 0;
}

//Duplication of the  just readed source chunk to the destiny image struct chunk
int duplicateImageChunk(ImagenData src, ImagenData dst, int dim){
    int i=0;
    
    for(i=0;i<dim;i++){
        dst->R[i] = src->R[i];
        dst->G[i] = src->G[i];
        dst->B[i] = src->B[i];
    }
//    printf ("Duplicated = %d pixels\n",i);
    return 0;
}

// Open kernel file and reading kernel matrix. The kernel matrix 2D is stored in 1D format.
kernelData leerKernel(char* nombre){
    FILE *fp;
    int i=0;
    kernelData kern=NULL;
    
    /*Opening the kernel file*/
    fp=fopen(nombre,"r");
    if(!fp){
        perror("Error: ");
    }
    else{
        //Memory allocation
        kern=(kernelData) malloc(sizeof(struct structkernel));
        
        //Reading kernel matrix dimensions
        fscanf(fp,"%d,%d,", &kern->kernelX, &kern->kernelY);
        kern->vkern = (float *)malloc(kern->kernelX*kern->kernelY*sizeof(float));
        
        // Reading kernel matrix values
        for (i=0;i<(kern->kernelX*kern->kernelY)-1;i++){
            fscanf(fp,"%f,",&kern->vkern[i]);
        }
        fscanf(fp,"%f",&kern->vkern[i]);
        fclose(fp);
    }
    return kern;
}

// Open the image file with the convolution results
int initfilestore(ImagenData img, FILE **fp, char* nombre, long *position){
    /*Se crea el fichero con la imagen resultante*/
    if ( (*fp=fopen(nombre,"w")) == NULL ){
        perror("Error: ");
        return -1;
    }
    /*Writing Image Header*/
    fprintf(*fp,"P%d\n%s\n%d %d\n%d\n",img->P,img->comentario,img->ancho,img->altura,img->maxcolor);
    *position = ftell(*fp);
    return 0;
}

// Writing the image partition to the resulting file. dim is the exact size to write. offset is the displacement for avoid halos.
int savingChunk(ImagenData img, FILE **fp, int dim, int offset){
    int i,k=0;
    //Writing image partition
    for(i=offset;i<dim+offset;i++){
        fprintf(*fp,"%d %d %d ",img->R[i],img->G[i],img->B[i]);
//        if ((i+1)%6==0) fprintf(*fp,"\n");
        k++;
    }
//    printf ("Writed = %d pixels, dim=%d, offset=%d\n",k,dim, offset);
    return 0;
}

// This function free the space allocated for the image structure.
void freeImagestructure(ImagenData *src){
    
    free((*src)->comentario);
    free((*src)->R);
    free((*src)->G);
    free((*src)->B);
    
    free(*src);
}

///////////////////////////////////////////////////////////////////////////////
// 2D convolution
// 2D data are usually stored in computer memory as contiguous 1D array.
// So, we are using 1D array for 2D data.
// 2D convolution assumes the kernel is center originated, which means, if
// kernel size 3 then, k[-1], k[0], k[1]. The middle of index is always 0.
// The following programming logics are somewhat complicated because of using
// pointer indexing in order to minimize the number of multiplications.
//
//
// signed integer (32bit) version:
///////////////////////////////////////////////////////////////////////////////
int convolve2D(int* in, int* out, int dataSizeX, int dataSizeY,
               float* kernel, int kernelSizeX, int kernelSizeY)
{
    int kCenterX, kCenterY;

    // check validity of params
    if(!in || !out || !kernel) return -1;
    if(dataSizeX <= 0 || kernelSizeX <= 0) return -1;
    
    // find center position of kernel (half of kernel size)
    kCenterX = (int)kernelSizeX / 2;
    kCenterY = (int)kernelSizeY / 2;
    
    // init working  pointers
    // inPtr = inPtr2 = &in[dataSizeX * kCenterY + kCenterX];  // note that  it is shifted (kCenterX, kCenterY),
    // omp directive here
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < dataSizeY; ++i)
    for (int j = 0; j < dataSizeX; ++j)
    {
        int rowMax = i + kCenterY;
        int rowMin = i - dataSizeY + kCenterY;

        int colMax = j + kCenterX;
        int colMin = j - dataSizeX + kCenterX;

        int sum = 0;

        for (int m = 0; m < kernelSizeY; ++m)
        {
            if (m <= rowMax && m > rowMin)
            {
                for (int n = 0; n < kernelSizeX; ++n)
                {
                    if (n <= colMax && n > colMin)
                    {
                        int input_row = i + (kCenterY - m);
                        int input_col = j + (kCenterX - n);

                        sum += in[input_row * dataSizeX + input_col] *
                            kernel[m * kernelSizeX + n];
                    }
                }
            }
        }

        if (sum >= 0)
            out[i * dataSizeX + j] = (int)(sum + 0.5f);
        else
            out[i * dataSizeX + j] = (int)(sum - 0.5f);
    }
    
    return 0;
}

void print_usage(const char* program_name){
    printf("Usage: %s <image-file> <kernel-file> <result-file> <partitions>\n", program_name);
    printf("\n\nError, Missing parameters:\n");
    printf("format: ./serialconvolution image_file kernel_file result_file\n");
    printf("- image_file : source image path (*.ppm)\n");
    printf("- kernel_file: kernel path (text file with 1D kernel matrix)\n");
    printf("- result_file: result image path (*.ppm)\n");
    printf("- partitions : Image partitions\n\n");
}

#define WORK_TAG 1
#define RESULT_TAG 2
#define TERMINATE_TAG 3


int master(int argc, char **argv) {
    if(argc != 5)
    {
        print_usage(argv[0]);
        return -1;
    }

    int partition_count = atoi(argv[4]);
    int partition_size, chunk_size, halo, halo_size;
    long file_position = 0;
    double start, tstart=0, tend=0, tread=0, tcopy=0, tconv=0, tstore=0, treadk=0;
    FILE *source_file=NULL,*result_file=NULL;
    ImagenData source_image=NULL, output_image=NULL;

    start = omp_get_wtime();
    tstart = start;
    kernelData kernel_data=NULL;
    if ( (kernel_data = leerKernel(argv[2]))==NULL) {
        return -1;
    }
    if (partition_count==1) halo=0;
    else halo = (kernel_data->kernelY/2)*2;
    treadk = treadk + (omp_get_wtime() - start);

    start = omp_get_wtime();
    if ( (source_image = initimage(argv[1], &source_file, partition_count, halo)) == NULL) {
        return -1;
    }
    tread = tread + (omp_get_wtime() - start);

    start = omp_get_wtime();
    if ( (output_image = duplicateImageData(source_image, partition_count, halo)) == NULL) {
        return -1;
    }
    tcopy = tcopy + (omp_get_wtime() - start);

    start = omp_get_wtime();
    if (initfilestore(output_image, &result_file, argv[3], &file_position)!=0) {
        perror("Error: ");
        return -1;
    }
    tstore = tstore + (omp_get_wtime() - start);

    int next_partition_to_assign = 0;
    partition_size  = (source_image->altura*source_image->ancho)/partition_count;
    MPI_Status status;
    char message_tag[1];
    int worker_rank;
    int mpi_world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
    int worker_count = mpi_world_size - 1;
    int terminated_workers = 0;
    int *worker_output_offsets = (int *)calloc(mpi_world_size, sizeof(int));
    int *worker_pixel_counts = (int *)calloc(mpi_world_size, sizeof(int));
    if (worker_count <= 0 || worker_output_offsets == NULL || worker_pixel_counts == NULL) {
        free(worker_output_offsets);
        free(worker_pixel_counts);
        return -1;
    }

    // Broadcast kernel data to all workers
    MPI_Bcast(&kernel_data->kernelX, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&kernel_data->kernelY, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(kernel_data->vkern, kernel_data->kernelX * kernel_data->kernelY, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Broadcast image dimensions to all workers
    MPI_Bcast(&source_image->ancho, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&source_image->altura, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&source_image->maxcolor, 1, MPI_INT, 0, MPI_COMM_WORLD);

    while (terminated_workers < worker_count) {
        // Read MPI_Recv get type of message and source
        MPI_Recv(message_tag, 1, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        worker_rank = status.MPI_SOURCE;
        
        if(message_tag[0] == WORK_TAG) {
            if (next_partition_to_assign < partition_count) {
                int partition_index = next_partition_to_assign++;
                if (partition_index==0) {
                    halo_size  = halo/2;
                    chunk_size = partition_size + (source_image->ancho*halo_size);
                    worker_output_offsets[worker_rank] = 0;
                }
                else if(partition_index<partition_count-1) {
                    halo_size  = halo;
                    chunk_size = partition_size + (source_image->ancho*halo_size);
                    worker_output_offsets[worker_rank] = (source_image->ancho*halo/2);
                }
                else {
                    halo_size  = halo/2;
                    chunk_size = partition_size + (source_image->ancho*halo_size);
                    worker_output_offsets[worker_rank] = (source_image->ancho*halo/2);
                }

                if (readImage(source_image, &source_file, chunk_size, halo/2, &file_position)) {
                    return -1;
                }
                tread = tread + (omp_get_wtime() - start);

                start = omp_get_wtime();
                if ( duplicateImageChunk(source_image, output_image, chunk_size) ) {
                    return -1;
                }
                // Send the chunk to the worker
                int chunk_height = (source_image->altura/partition_count)+halo_size;
                int pixel_count = source_image->ancho * chunk_height;
                worker_pixel_counts[worker_rank] = pixel_count;
                char assign_work = WORK_TAG;
                MPI_Send(&assign_work, 1, MPI_CHAR, worker_rank, WORK_TAG, MPI_COMM_WORLD);
                MPI_Send(&chunk_height, 1, MPI_INT, worker_rank, WORK_TAG, MPI_COMM_WORLD);
                MPI_Send(source_image->R, pixel_count, MPI_INT, worker_rank, WORK_TAG, MPI_COMM_WORLD);
                MPI_Send(source_image->G, pixel_count, MPI_INT, worker_rank, WORK_TAG, MPI_COMM_WORLD);
                MPI_Send(source_image->B, pixel_count, MPI_INT, worker_rank, WORK_TAG, MPI_COMM_WORLD);
            } else {
                char terminate_worker = TERMINATE_TAG;
                MPI_Send(&terminate_worker, 1, MPI_CHAR, worker_rank, TERMINATE_TAG, MPI_COMM_WORLD);
                terminated_workers++;
            }
        } else if(message_tag[0] == RESULT_TAG) {
            // save RGB channels to the output image struct
            int pixel_count = worker_pixel_counts[worker_rank];
            MPI_Recv(output_image->R, pixel_count, MPI_INT, worker_rank, RESULT_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(output_image->G, pixel_count, MPI_INT, worker_rank, RESULT_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(output_image->B, pixel_count, MPI_INT, worker_rank, RESULT_TAG, MPI_COMM_WORLD, &status);

            if (savingChunk(output_image, &result_file, partition_size, worker_output_offsets[worker_rank])) {
                perror("Error: ");
                return -1;
            }

        }
    }
    free(worker_output_offsets);
    free(worker_pixel_counts);

    fclose(source_file);
    fclose(result_file);

    tend = omp_get_wtime();

    printf("Imatge: %s\n", argv[1]);
    printf("ISizeX : %d\n", source_image->ancho);
    printf("ISizeY : %d\n", source_image->altura);
    printf("kSizeX : %d\n", kernel_data->kernelX);
    printf("kSizeY : %d\n", kernel_data->kernelY);
    printf("%.6lf seconds elapsed for Reading image file.\n", tread);
    printf("%.6lf seconds elapsed for copying image structure.\n", tcopy);
    printf("%.6lf seconds elapsed for Reading kernel matrix.\n", treadk);
    printf("%.6lf seconds elapsed for make the convolution.\n", tconv);
    printf("%.6lf seconds elapsed for writing the resulting image.\n", tstore);
    printf("%.6lf seconds elapsed\n", tend-tstart);

    freeImagestructure(&source_image);
    freeImagestructure(&output_image);

    return 0;
}

int worker(int argc, char **argv) {
    (void)argc;
    (void)argv;
    char message_tag[1];
    MPI_Status status;

    // Receive kernel data from master
    kernelData kernel_data = (kernelData) malloc(sizeof(struct structkernel));
    if (kernel_data == NULL) {
        return -1;
    }
    MPI_Bcast(&kernel_data->kernelX, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&kernel_data->kernelY, 1, MPI_INT, 0, MPI_COMM_WORLD);
    kernel_data->vkern = (float *)malloc(kernel_data->kernelX * kernel_data->kernelY * sizeof(float));
    if (kernel_data->vkern == NULL) {
        free(kernel_data);
        return -1;
    }
    MPI_Bcast(kernel_data->vkern, kernel_data->kernelX * kernel_data->kernelY, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Receive image dimensions from master
    int image_width, image_height, max_color;
    MPI_Bcast(&image_width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&image_height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_color, 1, MPI_INT, 0, MPI_COMM_WORLD);

    while(1){
        char request_work = WORK_TAG;
        MPI_Send(&request_work, 1, MPI_CHAR, 0, WORK_TAG, MPI_COMM_WORLD);
        MPI_Recv(message_tag, 1, MPI_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        
        if(message_tag[0] == TERMINATE_TAG) {
            break;
        } else if(message_tag[0] == WORK_TAG) {
            // Read data size from master
            int chunk_height;
            MPI_Recv(&chunk_height, 1, MPI_INT, 0, WORK_TAG, MPI_COMM_WORLD, &status);

            int pixel_count = image_width * chunk_height;
            int *source_r = (int *)malloc(sizeof(int) * pixel_count);
            int *source_g = (int *)malloc(sizeof(int) * pixel_count);
            int *source_b = (int *)malloc(sizeof(int) * pixel_count);
            int *result_r = (int *)malloc(sizeof(int) * pixel_count);
            int *result_g = (int *)malloc(sizeof(int) * pixel_count);
            int *result_b = (int *)malloc(sizeof(int) * pixel_count);
            if (source_r == NULL || source_g == NULL || source_b == NULL ||
                result_r == NULL || result_g == NULL || result_b == NULL) {
                free(source_r); free(source_g); free(source_b);
                free(result_r); free(result_g); free(result_b);
                free(kernel_data->vkern);
                free(kernel_data);
                return -1;
            }

            MPI_Recv(source_r, pixel_count, MPI_INT, 0, WORK_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(source_g, pixel_count, MPI_INT, 0, WORK_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(source_b, pixel_count, MPI_INT, 0, WORK_TAG, MPI_COMM_WORLD, &status);

            convolve2D(source_r, result_r, image_width, chunk_height, kernel_data->vkern, kernel_data->kernelX, kernel_data->kernelY);
            convolve2D(source_g, result_g, image_width, chunk_height, kernel_data->vkern, kernel_data->kernelX, kernel_data->kernelY);
            convolve2D(source_b, result_b, image_width, chunk_height, kernel_data->vkern, kernel_data->kernelX, kernel_data->kernelY);

            // Send the result back to master
            char result_ready = RESULT_TAG;
            MPI_Send(&result_ready, 1, MPI_CHAR, 0, RESULT_TAG, MPI_COMM_WORLD);
            MPI_Send(result_r, pixel_count, MPI_INT, 0, RESULT_TAG, MPI_COMM_WORLD);
            MPI_Send(result_g, pixel_count, MPI_INT, 0, RESULT_TAG, MPI_COMM_WORLD);
            MPI_Send(result_b, pixel_count, MPI_INT, 0, RESULT_TAG, MPI_COMM_WORLD);

            free(source_r); free(source_g); free(source_b);
            free(result_r); free(result_g); free(result_b);
        }
    }

    free(kernel_data->vkern);
    free(kernel_data);
    
    return 0;
}


//////////////////////////////////////////////////////////////////////////////////////////////////
// MAIN FUNCTION
//////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    int mpi_rank, mpi_world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
    (void)mpi_world_size;
    int process_exit_code;
    if (mpi_rank == 0) {
        process_exit_code = master(argc, argv);
    } else {
        process_exit_code = worker(argc, argv);
    }

    MPI_Finalize();

    return process_exit_code;
}
