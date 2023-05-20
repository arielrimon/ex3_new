#include <atomic>
#include <csignal>
#include <condition_variable>
#include <algorithm>
#include <unistd.h>
#include <malloc.h>
#include "MapReduceFramework.h"
#include "Barrier.h"


///------------------------------ STRUCTS -----------------------------------------
//struct JobHandleObject {
//    JobHandleObject(JobState *job_state, pthread_t *threads, int number_of_threads) {
//        this->job_state = job_state;
//        this->threads = threads;
//        this->number_of_threads = number_of_threads;
//        this->flag.store(0);
//    }
//
//    JobState *job_state;
//    std::atomic<int> flag;
//    pthread_t *threads;
//    std::mutex wait_mutex;
//    int number_of_threads;
//
//    void activate_flag() {
//        this->flag.store(1);
//    }
//
//    bool get_flag() {
//        return this->flag.load();
//    }
//};

struct JobContext { // resources used by all threads - every thread hold a pointer
    pthread_t *threads;
    JobState *job_state;
    int number_of_threads;
    std::vector<IntermediateVec *> *all_intermediate_vec;
    std::vector<IntermediateVec *> *shuffled_intermediate_vec;
    const MapReduceClient *client;
    OutputVec *output_vec;
    std::mutex *end_map_stage_mutex; //todo: fix
    std::atomic<int> *map_atomic_counter;
    std::atomic<int> *reduce_atomic_counter;
    Barrier *barrier;
    std::atomic<int> flag;
    std::mutex wait_mutex;
    float total_items_to_process;
    std::atomic<int> *current_processed;

    void activate_flag() {
        this->flag.store(1);
    }

    bool get_flag() {
        return this->flag.load();
    }

//
//
//    JobContext(JobState *job_state, pthread_t *threads, int number_of_threads) {
//        this->job_state =job_state;
//        this->threads = threads;
//        this->number_of_threads = number_of_threads;
//        this->flag.store(0);
//        this->client = &client;
//        this->all_intermediate_vec = all_vectors;
//        this->shuffled_intermediate_vec = shuffled_intermediate_vec;
//        this->output_vec = &outputVec;
//        this->end_map_stage_mutex = (std::mutex *) malloc(sizeof(std::mutex));
//        this->job_state = job_state;
//        this->barrier = new Barrier(numOfThreads);
//        this->map_atomic_counter = map_atomic_counter;
//        this->reduce_atomic_counter = reduce_atomic_counter;
//    }

};

struct ThreadContext {
    int thread_id;
    const InputVec *input_vec;
    JobContext *job_context; // JobContext of all threads
    IntermediateVec *intermediate_vec;
};

///------------------------------ FUNCTIONS USED -----------------------------------------

// init funcs
JobState *get_new_job_state();

ThreadContext *init_thread_contexts(OutputVec &outputVec, int multiThreadLevel, const MapReduceClient &client,
                                    const InputVec &inputVec, JobState *job_state, JobContext *job_context);

void init_thread_context(const InputVec &inputVec, ThreadContext *thread_context, JobContext *job_context, int i);

JobContext *
init_job_context(OutputVec &outputVec, const MapReduceClient &client, JobState *job_state, int numOfThreads);

// helpers for shuffle funcs
K2 *get_max_key(std::vector<IntermediateVec *> *all_vectors);

void
pop_all_max_keys(K2 *max_key, IntermediateVec *intermediateVecOutput, std::vector<IntermediateVec *> *all_vec_input,
                 ThreadContext *tc, int &current_number_of_processed_pairs);

void remove_empty_vectors(std::vector<IntermediateVec *> *all_vectors, ThreadContext *pContext);


void release_all_resources(JobHandle job_handle);

void update_stage(JobContext* job_context, stage_t stage, float i);

int get_size_of_vector_of_vectors(std::vector<IntermediateVec *> *pVector);

/**
 *
 * @param a IntermediatePair
 * @param b IntermediatePair
 * @return true if (key in a) < (key in b)
 */
bool compare(const IntermediatePair &a, const IntermediatePair &b) {
    return *(a.first) < *(b.first);
}

// map reduce
void *map_reduce_method(void *context) {
    auto *tc = (ThreadContext *) context;

    //Map
    auto input_vec_size = (float)tc->input_vec->size();
    update_stage(tc->job_context, MAP_STAGE, input_vec_size);
    int current_index = (*(tc->job_context->map_atomic_counter))++;
    tc->job_context->total_items_to_process = input_vec_size;
    while (current_index < input_vec_size) {
        //Todo: how should I increase the percentage
        tc->job_context->client->map(tc->input_vec->at(current_index).first, tc->input_vec->at(current_index).second,
                                     context);
        (*(tc->job_context->current_processed))++;
//        tc->job_context->job_state->percentage = ((float) (current_index + 1) / (float) input_vec_size) * 100;
        current_index = (*(tc->job_context->map_atomic_counter))++;
    }
    //Sort
    std::sort(tc->intermediate_vec->begin(), tc->intermediate_vec->end(), compare);
    tc->job_context->barrier->barrier();

    //Shuffle
    int all_intermediate_vec_size = get_size_of_vector_of_vectors(tc->job_context->all_intermediate_vec);
    update_stage(tc->job_context, SHUFFLE_STAGE, all_intermediate_vec_size);
    if (tc->thread_id == 0) {
        std::vector<IntermediateVec *> *all_intermediate_vec = tc->job_context->all_intermediate_vec;
        std::vector<IntermediateVec *> *shuffled_vector = tc->job_context->shuffled_intermediate_vec;
//        int current_number_of_processed_pairs = 0;
        while (!all_intermediate_vec->at(0)->empty()) {
            K2 *max_key = get_max_key(all_intermediate_vec); // gets the maximal key of all keys imn all vec
            auto *max_key_vector = (IntermediateVec *) malloc(sizeof(IntermediateVec));
            pop_all_max_keys(max_key, max_key_vector, all_intermediate_vec, tc, all_intermediate_vec_size);
            shuffled_vector->push_back(max_key_vector);
        }

    }
    tc->job_context->barrier->barrier();
    //Reduce
    float number_of_shuffled_items = get_size_of_vector_of_vectors(tc->job_context->shuffled_intermediate_vec);
    update_stage(tc->job_context, REDUCE_STAGE, number_of_shuffled_items);
    current_index = (*(tc->job_context->reduce_atomic_counter))++;

    float current_number_of_reduced_items = 0;
    while (current_index < tc->job_context->shuffled_intermediate_vec->size()) {
        current_number_of_reduced_items += tc->job_context->shuffled_intermediate_vec->at(current_index)->size();
        tc->job_context->client->reduce(tc->job_context->shuffled_intermediate_vec->at(current_index), context);
        tc->job_context->job_state->percentage = (current_number_of_reduced_items  / number_of_shuffled_items) * 100;
        current_index = (*(tc->job_context->reduce_atomic_counter))++;
    }
    return tc->job_context->job_state;
}

int get_size_of_vector_of_vectors(std::vector<IntermediateVec *> *pVector) {
    int sum = 0;
    for (const auto &innerVec: *pVector) {
        sum += innerVec->size();
    }
    return sum;
}

//void update_stage(JobState *job_stage, stage_t stage) {
//
//    job_stage->stage = stage;
//    job_stage->percentage = 0;
//}
void update_stage(JobContext* jc, stage_t stage, float total) {

    //TODO: ADD MUTEX as these are many atomic functions together
    jc->job_state->stage = stage;
    jc->job_state->percentage = 0;
    jc->total_items_to_process = total;
    jc->current_processed->store(0);
}

/**
 * goes over all vectors and checks only at the back cell for max value (as the cells are sorted)
 * @param all_vectors after sort phase
 * @return value of the maximal key found
 */
K2 *get_max_key(std::vector<IntermediateVec *> *all_vectors) {
    K2 *max_key = (K2 *) malloc(sizeof(K2));
    bool is_first_iteration = true;
    for (auto vec: *all_vectors) {
        if (is_first_iteration) {
            max_key = vec->back().first;
            is_first_iteration = false;
            continue;
        }
        max_key = std::max(max_key, vec->back().first);
    }
    return max_key;
}

/**
 *
 * @param max_key found after going over all vec.back()
 * @param intermediateVecOutput the output vector to place the result
 * @param all_vec_input all vectors
 */
void
pop_all_max_keys(K2 *max_key, IntermediateVec *intermediateVecOutput, std::vector<IntermediateVec *> *all_vec_input,
                 ThreadContext *tc, int &current_number_of_processed_pairs) {
    for (auto vec: *all_vec_input) {
        while (!vec->empty() &&
               vec->back().first == max_key) { // goes over the vector backwards as long as it equals the max_key
            intermediateVecOutput->push_back(vec->back());
            vec->pop_back();
            (*(tc->job_context->current_processed))++;
//            tc->job_context->job_state->percentage =
//                    ((float) current_number_of_processed_pairs / (float) (initial_size)) * 100;
        }
    }
    //after finishing emptying every vector from max_key pairs - going over and deleting all empty vectors
    remove_empty_vectors(all_vec_input, tc);
}

/**
 * frees the vec alloc when empty, and returns true. otherwise false without free
 * @param intermediate_vec out of all vectors - to be freed when empty
 * @return true if it was empty, otherwise false (without deleting)
 */
bool free_if_empty(const IntermediateVec *intermediate_vec) {
    bool flag = intermediate_vec->empty();
    if (flag) {
        delete intermediate_vec; //only if empty!
    }
    return flag;
}

/**
 * Remove empty vectors of all vectors
 * @param all_vectors after sort phase
 */
void remove_empty_vectors(std::vector<IntermediateVec *> *all_vectors, ThreadContext *pContext) {
    if (all_vectors->size() != 1) {
        bool (*check_if_vector_empty)(const IntermediateVec *) = [](const IntermediateVec *intermediate_vec) {
            return free_if_empty(
                    intermediate_vec);
        };
        const std::vector<IntermediateVec *>::iterator &vectors_to_erase = std::remove_if(all_vectors->begin(),
                                                                                          all_vectors->end(),
                                                                                          check_if_vector_empty);
        all_vectors->erase(
                vectors_to_erase, all_vectors->end());
    }
}

/**
 *
 * @param outputVec
 * @param multiThreadLevel num of threads to create
 * @param client
 * @param inputVec
 * @param job_state
 * @param job_context shared resources
 * @return
 */
ThreadContext *init_thread_contexts(OutputVec &outputVec, int multiThreadLevel, const MapReduceClient &client,
                                    const InputVec &inputVec, JobState *job_state, JobContext *job_context) {
    auto *thread_contexts = (ThreadContext *) malloc(multiThreadLevel * sizeof(ThreadContext));

    for (int i = 0; i < multiThreadLevel; ++i) {
        init_thread_context(inputVec, thread_contexts + i, job_context, i);
    }
    return thread_contexts;
}

/**
 *
 * @param inputVec of all pairs before map phase
 * @param thread_context
 * @param job_context pointer to shared resources
 * @param i index in threads
 */
void init_thread_context(const InputVec &inputVec, ThreadContext *thread_context,
                         JobContext *job_context, const int i) {
    thread_context->thread_id = i;
    thread_context->job_context = job_context;
    thread_context->input_vec = &inputVec;
    thread_context->intermediate_vec = new IntermediateVec();
    job_context->all_intermediate_vec->push_back(thread_context->intermediate_vec);
}

/**
 *
 * @param outputVec
 * @param client
 * @param job_state
 * @param numOfThreads
 * @return pointer to JobContext obj that holds all resources for all threads
 */
JobContext *
init_job_context(OutputVec &outputVec, const MapReduceClient &client, JobState *job_state, int numOfThreads) {
    // called only once as the resources are same for all threads
    //allocation of all resources
    auto *map_atomic_counter = (std::atomic<int> *) malloc(sizeof(std::atomic<int>));
    map_atomic_counter->store(0);
    auto *reduce_atomic_counter = (std::atomic<int> *) malloc(sizeof(std::atomic<int>));
    reduce_atomic_counter->store(0);
    auto *all_vectors = new std::vector<IntermediateVec *>();
    auto *shuffled_intermediate_vec = new std::vector<IntermediateVec *>();
    auto *job_context = (JobContext *) malloc(sizeof(JobContext));
    auto *current_processed = (std::atomic<int> *) malloc(sizeof(std::atomic<int>));
    current_processed->store(0);
    reduce_atomic_counter->store(0);

    // placing all resources in the JobContext obj
    job_context->client = &client;
    job_context->all_intermediate_vec = all_vectors;
    job_context->shuffled_intermediate_vec = shuffled_intermediate_vec;
    job_context->output_vec = &outputVec;
    job_context->end_map_stage_mutex = (std::mutex *) malloc(sizeof(std::mutex));
    job_context->job_state = job_state;
    job_context->barrier = new Barrier(numOfThreads);
    job_context->map_atomic_counter = map_atomic_counter;
    job_context->reduce_atomic_counter = reduce_atomic_counter;
    job_context->total_items_to_process = 0;
    job_context->current_processed = current_processed;

    return job_context;
}

/**
 *
 * @return JobState obj where stage = UNDEFINED_STAGE
 */
JobState *get_new_job_state() {
    auto *job = (JobState *) malloc(sizeof(JobState));
    job->stage = UNDEFINED_STAGE;
    job->percentage = 0;
    return job;
}


///------------------------------ LIBRARY -----------------------------------------

/**
 * starts running the MapReduce algorithm (w several threads) and returns a job handler
 * You can assume that the input to this function is valid
 * @param client the task that the framework should run (The implementation of MapReduceClient)
 * @param inputVec a vector of type std::vector<std::pair<K1*, V1*>>, the input elements
 * @param outputVec a vector of type std::vector<std::pair<K3*, V3*>>,
 *                  to which the output elements will be added before returning.
 *                  You can assume that outputVec is empty.
 * @param multiThreadLevel the number of worker threads to be used for running the algorithm
 *                  You can assume multiThreadLevel argument is valid (greater or equal to 1)
 * @return JobHandle that will be used for monitoring the job
 */
JobHandle
startMapReduceJob(const MapReduceClient &client, const InputVec &inputVec, OutputVec &outputVec, int multiThreadLevel) {
    pthread_t threads[multiThreadLevel];
    JobState *job_state = get_new_job_state();
    // NOTICE: I changed it so we build it only once and every thread gets a pointer of it! works the same!
    JobContext *job_context = init_job_context(outputVec, client, job_state, multiThreadLevel);

    ThreadContext *thread_contexts = init_thread_contexts(outputVec, multiThreadLevel, client, inputVec, job_state,
                                                          job_context);

    for (int i = 0; i < multiThreadLevel; ++i) {
        pthread_create(threads + i, nullptr, map_reduce_method, thread_contexts + i);
    }


//
//    int counter = 0;
//    for (const auto &pair: inputVec) {
//        client.map(pair.first, pair.second, thread_contexts + counter);
//        counter++;
//    }

    return static_cast<JobHandle>(job_context);
}

/**
 * gets a JobHandle and updates the state of the job into the given JobState struct
 * @param job JobHandle
 * @param state JobState struct to update the job to
 */
void getJobState(JobHandle job, JobState *state) {
    auto *job_context = static_cast<JobContext *>(job);
    state->stage = job_context->job_state->stage;
    //TODO: put mutex here
    state->percentage = ((float) job_context->current_processed->load()/job_context->total_items_to_process)*100;
}

/**
 * Releasing all resources of a job.
 * After this function is called the job handle will be invalid.
 * In case that the function is called and the job is not finished yet wait until the job is finished to close it
 * @param job JobHandle
 */
void closeJobHandle(JobHandle job) {
    waitForJob(job);
    release_all_resources(job);
}

void release_all_resources(JobHandle job_handle) {
    auto *job_context = static_cast<JobContext *>(job_handle);
}

/**
 * produces a (K2*, V2*) pair.
 * updates the number of intermediary elements using atomic counter
 * @param key of input intermediary element
 * @param value of input intermediary element
 * @param context data structure of the thread that created the intermediary element
 */
void emit2(K2 *key, V2 *value, void *context) {
    auto *tc = (ThreadContext *) context;
    IntermediatePair pair = IntermediatePair(key, value);
    tc->intermediate_vec->push_back(pair);
}

/**
 * produces a (K3*, V3*) pair.
 * saves the output element in the context data structures
 * updates the number of output elements using atomic counter
 * @param key of input output element
 * @param value of input output element
 * @param context data structure of the thread that created the output element
 */
void emit3(K3 *key, V3 *value, void *context) {
    auto *tc = (ThreadContext *) context;
    OutputPair pair = OutputPair(key, value);
    tc->job_context->output_vec->push_back(pair);
}

/**
 * gets the JobHandle returned by startMapReduceFramework and waits until it is finished
 * @param job the JobHandle returned by startMapReduceFramework
 */
void waitForJob(JobHandle job) {
    auto *job_context = static_cast<JobContext *>(job);
    job_context->wait_mutex.lock();
    if (!job_context->get_flag()) {
        job_context->activate_flag();
        for (int i = 0; i < job_context->number_of_threads; ++i) {
            pthread_join(job_context->threads[i], nullptr);
        }
    }
    job_context->wait_mutex.unlock();
};