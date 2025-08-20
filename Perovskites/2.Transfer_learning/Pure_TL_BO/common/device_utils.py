import torch
import sys

def get_optimal_device(prefer_gpu: bool = True, verbose: bool = True) -> str:
    """
    최적의 PyTorch 디바이스를 자동으로 선택합니다.
    
    우선순위: CUDA > MPS > CPU
    
    Args:
        prefer_gpu: GPU 사용을 선호할지 여부
        verbose: 디바이스 정보 출력 여부
        
    Returns:
        디바이스 문자열 ('cuda', 'mps', 'cpu')
    """
    if not prefer_gpu:
        if verbose:
            print("🖥️  GPU 사용 비활성화 - CPU 사용")
        return 'cpu'
    
    # CUDA 확인 (NVIDIA GPU)
    if torch.cuda.is_available():
        device = 'cuda'
        if verbose:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🚀 CUDA GPU 감지: {gpu_name} ({gpu_memory:.1f}GB)")
        return device
    
    # MPS 확인 (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        if verbose:
            # Apple Silicon 정보 확인
            if sys.platform == 'darwin':
                try:
                    import subprocess
                    result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                          capture_output=True, text=True)
                    cpu_info = result.stdout.strip()
                    print(f"🍎 MPS (Apple Silicon) 감지: {cpu_info}")
                except:
                    print("🍎 MPS (Apple Silicon) 사용 가능")
            else:
                print("🍎 MPS 사용 가능")
        return device
    
    # CPU 폴백
    if verbose:
        print("🖥️  GPU 없음 - CPU 사용")
    return 'cpu'


def check_device_compatibility(device: str, verbose: bool = True) -> bool:
    """
    선택된 디바이스의 호환성을 확인합니다.
    
    Args:
        device: 확인할 디바이스
        verbose: 상세 정보 출력 여부
        
    Returns:
        호환성 여부
    """
    try:
        # 간단한 텐서 연산으로 디바이스 테스트
        test_tensor = torch.randn(2, 2, device=device)
        result = torch.mm(test_tensor, test_tensor)
        
        if verbose:
            print(f"✅ {device.upper()} 디바이스 정상 동작 확인")
        return True
        
    except Exception as e:
        if verbose:
            print(f"❌ {device.upper()} 디바이스 오류: {e}")
            print("   CPU로 폴백합니다.")
        return False


def setup_device_for_bnn(prefer_gpu: bool = True, verbose: bool = True) -> str:
    """
    BNN을 위한 최적 디바이스 설정
    
    Args:
        prefer_gpu: GPU 사용 선호 여부
        verbose: 상세 정보 출력 여부
        
    Returns:
        사용할 디바이스 문자열
    """
    if verbose:
        print("🔍 BNN을 위한 최적 디바이스 검색 중...")
        
    device = get_optimal_device(prefer_gpu=prefer_gpu, verbose=verbose)
    
    # 디바이스 호환성 확인
    if not check_device_compatibility(device, verbose=False):
        if verbose:
            print(f"⚠️  {device.upper()} 호환성 문제 - CPU로 폴백")
        device = 'cpu'
        check_device_compatibility(device, verbose=verbose)
    
    # MPS 특별 권장사항
    if device == 'mps' and verbose:
        print("💡 MPS 사용 팁:")
        print("   - Apple Silicon에서 상당한 성능 향상 예상")
        print("   - 메모리 사용량이 CUDA와 다를 수 있습니다")
        print("   - 문제 발생 시 --device cpu 옵션 사용")
    
    return device


def get_device_memory_info(device: str) -> dict:
    """
    디바이스 메모리 정보 반환
    
    Args:
        device: 디바이스 문자열
        
    Returns:
        메모리 정보 딕셔너리
    """
    info = {'device': device}
    
    if device == 'cuda' and torch.cuda.is_available():
        info['total_memory'] = torch.cuda.get_device_properties(0).total_memory
        info['allocated_memory'] = torch.cuda.memory_allocated(0)
        info['cached_memory'] = torch.cuda.memory_reserved(0)
    elif device == 'mps':
        # MPS는 시스템 메모리를 공유하므로 정확한 정보 제공이 어려움
        info['note'] = 'MPS uses shared system memory'
    else:
        info['note'] = 'CPU uses system RAM'
    
    return info


# MPS 관련 유틸리티 함수들
def clear_mps_cache():
    """MPS 캐시 정리 (메모리 최적화)"""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def set_mps_optimizations():
    """MPS 최적화 설정"""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS 관련 최적화 설정이 있다면 여기에 추가
        pass