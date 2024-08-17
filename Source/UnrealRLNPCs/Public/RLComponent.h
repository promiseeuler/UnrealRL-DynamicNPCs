#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "RLComponent.generated.h"

UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class UNREALRLNPCS_API URLComponent : public UActorComponent
{
    GENERATED_BODY()

public:    
    URLComponent();

    virtual void BeginPlay() override;
    virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

    UFUNCTION(BlueprintCallable, Category = "Reinforcement Learning")
    void PerformAction(int32 ActionIndex);

    UFUNCTION(BlueprintCallable, Category = "Reinforcement Learning")
    void ReceiveReward(float Reward);

protected:
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Reinforcement Learning")
    TArray<FString> ActionSpace;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Reinforcement Learning")
    int32 StateSize;

private:
    // Add private members as needed
};