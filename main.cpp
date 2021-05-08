#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <openmpi/ompi/mpi/cxx/mpicxx.h>

#define MACHINE_ROOT_ID 0
#define MSG_SORT_TAG    0

#define DATA_TYPE       std::uint32_t
#define SEND_TYPE       MPI_UINT32_T

// Storage info about Machine and clusters
struct Machine {

    static bool IsMaster() noexcept {
        return m_MachineID == MACHINE_ROOT_ID;
    }

    inline static int m_MachineID;      // Current process machine ID
    inline static int m_MachineSize;    // Count of concurrent machines

    inline static constexpr std::uint32_t m_SizeArray = 100;
};

// Fill array with random values
template <typename Type, typename = std::enable_if_t<std::is_integral_v<Type>>>
void Fill(std::vector<Type>& array) {
    std::random_device random_device;
    std::mt19937 random_engine(random_device());
    std::uniform_int_distribution<int> distribution(1, array.size());

    for(auto& value : array) {
        value = distribution(random_engine);
    }
}

// Sort array with algorithm bubble sort
template <typename Type, typename = std::enable_if_t<std::is_integral_v<Type>>>
void Sort(std::vector<Type>& array)
{
    for(Type i = 0; i < array.size(); ++i)
    {
        bool isFinish = true;
        for(Type j = 0; j < array.size() - (i + 1); ++j)
        {
            if(array[j] > array[j + 1]) {
                isFinish = false;
                std::swap(array[j], array[j + 1]);
            }
        }
        
        if(isFinish) {
            break;
        }
    }
}

// Entry point for Master machine
// If no cluster machines, then the Master machine will sort it
// Otherwise, the array is split and sent to clusters for sorting
void Master()
{
    // Prepare array and fill with random values
    std::vector<DATA_TYPE> array;
    array.resize(Machine::m_SizeArray);
    Fill(array);

    for(auto value : array) {
        std::cout << "[Master] ID: " << Machine::m_MachineID << " fill value: " << value << " to array" << std::endl;
    }

    // Check count of machines
    if(Machine::m_MachineSize == 1)
    {
        std::cout << "[Info] Cluster machines not found, using Master machine for sorting!" << std::endl;
        std::cout << "----- [Start sorting with Master] -----" << std::endl;
        Sort(array);
        for(auto value : array) {
            std::cout << "[Master] ID: " << Machine::m_MachineID << " sorted value: " << value << std::endl;
        }

    } else {
        std::cout << std::endl;
        std::cout << "----- [Start sorting with Clusters: " << (Machine::m_MachineSize - 1) << "] -----" << std::endl;

        // Calculate distribution of array
        const std::uint32_t taskPerProcess = static_cast<std::uint32_t>(array.size()) / (Machine::m_MachineSize - 1);
        const std::uint32_t taskRemainder = static_cast<std::uint32_t>(array.size()) - (Machine::m_MachineSize - 1) * taskPerProcess;

        std::cout << "[Info] TaskPerProcess: " << taskPerProcess << " | taskRemainder: " << taskRemainder << std::endl;

        // Send task to Clusters for sorting
        std::uint32_t offset = 0;
        for(int i = 1; i < Machine::m_MachineSize; ++i)
        {
            const bool isLast = (i == (Machine::m_MachineSize - 1));

            if(isLast) {
                std::cout << "[debug] send size: " << taskPerProcess + taskRemainder << std::endl;
                MPI_Send(&array[offset], taskPerProcess + taskRemainder, SEND_TYPE, i, MSG_SORT_TAG, MPI_COMM_WORLD);
            } else {
                std::cout << "[debug] send size: " << taskPerProcess << std::endl;
                MPI_Send(&array[offset], taskPerProcess, SEND_TYPE, i, MSG_SORT_TAG, MPI_COMM_WORLD);
            }

            offset += taskPerProcess;
        }

        std::cout << std::endl;
        for(auto value : array) {
            std::cout << "[Master] ID: " << Machine::m_MachineID << " before sort: " << value << std::endl;
        }
        std::cout << std::endl;

        // -------- [Recv from clusters]
        // Initialize MPI Status and Vector for storage sorted data from clusters
        MPI_Status status;
        std::vector<std::vector<DATA_TYPE>> data;
        data.resize(Machine::m_MachineSize - 1);

        // Recv sorted array from clusters and fill data array's
        for(int i = 1; i < Machine::m_MachineSize; ++i)
        {
            offset = i - 1;
            const bool isLast = (i == (Machine::m_MachineSize - 1));

            if(isLast) {
                data[offset].resize(taskPerProcess + taskRemainder);
            } else {
                data[offset].resize(taskPerProcess);
            }

            MPI_Recv(data[offset].data(), data[offset].size(), SEND_TYPE, i, MSG_SORT_TAG, MPI_COMM_WORLD, &status);

            if(status.MPI_ERROR) {
                std::cout << "[Master] ID: " << Machine::m_MachineID << " recv data from ClusterID: " << i<< " error: " << status.MPI_ERROR << std::endl;
                continue;
            }

            std::int32_t dataSize = 0;
            MPI_Get_count(&status, SEND_TYPE, &dataSize);

            std::cout << "[Master] ID: " << Machine::m_MachineID << " recv size: " << dataSize << " from ClusterID: " << status.MPI_SOURCE << std::endl;
        }

        // Output sorted data
        // Sorted data storage in "data", where offset 0 - it's cluster offset
        std::cout << std::endl;
        std::cout << "----------[Results]----------" << std::endl;
        for(std::size_t i = 0; i < data.size(); ++i)
        {
            const auto& vec = data[i];
            std::cout << "--[Result from Cluster ID: " << i + 1 << "]--" << std::endl;
            for(auto value : vec) {
                std::cout << "Sorted value: " << value << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

// Entry point for Cluster machine
void Cluster() {
    MPI_Status status;
    std::vector<DATA_TYPE> array;

    {
        // Calculate size of array
        const bool isLast = (Machine::m_MachineID == (Machine::m_MachineSize - 1));
        const std::uint32_t taskPerProcess = Machine::m_SizeArray / (Machine::m_MachineSize - 1);
        const std::uint32_t taskRemainder = Machine::m_SizeArray - (Machine::m_MachineSize - 1) * taskPerProcess;

        if(isLast) {
            array.resize(taskPerProcess + taskRemainder);
        } else {
            array.resize(taskPerProcess);
        }
    }

    // Recv array from Master machine and fill in array
    MPI_Recv(array.data(), array.size(), SEND_TYPE, MACHINE_ROOT_ID, MSG_SORT_TAG, MPI_COMM_WORLD, &status);

    std::cout << std::endl;
    if(status.MPI_ERROR) {
        std::cout << "[Cluster] ID: " << Machine::m_MachineID << " MPI_Recv error: " << status.MPI_ERROR << std::endl;
    }

    /*
    std::int32_t dataSize = 0;
    MPI_Get_count(&status, SEND_TYPE, &dataSize);

    std::cout << "[Cluster] ID: " << Machine::m_MachineID << " recv data size: " << dataSize << std::endl;

    for(auto value : array) {
        std::cout << "[Cluster] ID: " << Machine::m_MachineID << " recv data value: " << value << std::endl;
    }
    std::cout << std::endl; */

    Sort(array);
    /*
    for(auto value : array) {
        std::cout << "[Cluster] ID: " << Machine::m_MachineID << " sorted value: " << value << std::endl;
    }*/

    //std::cout << "[Cluster] ID: " << Machine::m_MachineID << " send data size: " << array.size() << " to MasterID: " << status.MPI_SOURCE << std::endl;
    
    // Send result on Master machine
    MPI_Send(array.data(), array.size(), SEND_TYPE, status.MPI_SOURCE, MSG_SORT_TAG, MPI_COMM_WORLD);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &Machine::m_MachineID);
    MPI_Comm_size(MPI_COMM_WORLD, &Machine::m_MachineSize);

    std::cout << (Machine::IsMaster() ? "[Master]" : "[Cluster]") << " ID: " << Machine::m_MachineID << " initialized";
    if(Machine::IsMaster() && Machine::m_MachineSize > 1) {
        std::cout << ", clusters: " << (Machine::m_MachineSize - 1);
    }
    std::cout << std::endl;

    if(Machine::IsMaster()) {
        Master();
    } else {
        Cluster();
    }

    MPI_Finalize();
    return 0;
}
