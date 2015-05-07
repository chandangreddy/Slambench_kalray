#include "mmu_lib.h"
#include "kernels.h"

/* Virtual memory start address */
extern int VIRT_U_MEM_PAG;


/* global pointers */
static uint16_t* inputDepth;
static uchar4* depthRender, trackRender, volumeRender;

// input once
static  float * gaussian;

// // inter-frame
static  Volume volume;
static  float3 * vertex;
static float3 * normal;

// // intra-frame
static TrackData * trackingResult;
static float* reductionoutput;
static float ** ScaledDepth;
static float * floatDepth;
static Matrix4 oldPose;
static Matrix4 raycastPose;
static float3 ** inputVertex;
static float3 ** inputNormal;


class Global_mem_mang   {
    int offset;
    static Global_mem_mang *s_instance;

    globla_mem_mang(){
        offset = 0;
    }

    public:
    void *malloc(int size){
        void *ptr = ((uintptr_t)&VIRT_U_MEM_PAG + offset + size);
        offset += size;
        return ptr;
    }

    void free(void *ptr){
    }

    static Global_mem_mang* instance(){
        if(!s_instance)
            s_instance = new Global_mem_mang();

         return s_instance;
    }
};

Global_mem_mang *Global_mem_mang::s_instance = 0;

void global_mem_init(){

inputDepth = (uint16_t*) Global_mem_mang::instance()->malloc(
                sizeof(uint16_t) * inputSize.x * inputSize.y);

depthRender = (uchar4*)  Global_mem_mang::instance()->malloc(
                    sizeof(uchar4) * computationSize.x * computationSize.y);

trackRender = (uchar4*) Global_mem_mang::instance()->malloc(
                        sizeof(uchar4) * computationSize.x * computationSize.y);

volumeRender = (uchar4*) Global_mem_mang::instance()->malloc(
                            sizeof(uchar4) * computationSize.x * computationSize.y);

}

void kfusion_global_mem_init(int iterations_size, int computationSize_x, int computationSize_y, size_t gaussianS){

    reductionoutput = (float*) Global_mem_mang::instance()->malloc(sizeof(float) * 8 * 32);

    ScaledDepth = (float**) Global_mem_mang::instance()->malloc(sizeof(float*) * iterations_size);
    inputVertex = (float3**) Global_mem_mang::instance()->malloc(sizeof(float3*) * iterations_size);
    inputNormal = (float3**) Global_mem_mang::instance()->malloc(sizeof(float3*) * iterations_size);

    for (unsigned int i = 0; i < iterations_size; ++i) {
        ScaledDepth[i] = (float*) Global_mem_mang::instance()->malloc(
                sizeof(float) * (computationSize_x * computationSize_y)
                        / (int) pow(2, i));
        inputVertex[i] = (float3*) Global_mem_mang::instance()->malloc(
                sizeof(float3) * (computationSize_x * computationSize_y)
                        / (int) pow(2, i));
        inputNormal[i] = (float3*) Global_mem_mang::instance()->malloc(
                sizeof(float3) * (computationSize_x * computationSize_y)
                        / (int) pow(2, i));
    }

    floatDepth = (float*) Global_mem_mang::instance()->malloc(
            sizeof(float) * computationSize_x * computationSize_y);
    vertex = (float3*) Global_mem_mang::instance()->malloc(
            sizeof(float3) * computationSize_x * computationSize_y);
    normal = (float3*) Global_mem_mang::instance()->malloc(
            sizeof(float3) * computationSize_x * computationSize_y);
    trackingResult = (TrackData*) Global_mem_mang::instance()->malloc(
            sizeof(TrackData) * computationSize_x * computationSize_y);

    gaussian = (float*) Global_mem_mang::instance()->malloc(gaussianS * sizeof(float), 1);

}
