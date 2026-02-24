use libc::{c_char, c_float, c_int, c_void};

// Type aliases matching libxsmm
type LibxsmmBlasint = c_int; // LP64: 32-bit int
type LibxsmmBitfield = libc::c_uint;

// ============================================================================
// Data types (from libxsmm_typedefs.h line 209)
// ============================================================================

pub const LIBXSMM_DATATYPE_F64: c_int = 0;
pub const LIBXSMM_DATATYPE_F32: c_int = 1;
pub const LIBXSMM_DATATYPE_BF16: c_int = 2;
pub const LIBXSMM_DATATYPE_F16: c_int = 3;
pub const LIBXSMM_DATATYPE_I8: c_int = 12;
pub const LIBXSMM_DATATYPE_U8: c_int = 13;

// ============================================================================
// GEMM flags (from libxsmm_typedefs.h line 448)
// ============================================================================

pub const LIBXSMM_GEMM_FLAG_NONE: LibxsmmBitfield = 0;
pub const LIBXSMM_GEMM_FLAG_BETA_0: LibxsmmBitfield = 4;
pub const LIBXSMM_GEMM_FLAG_VNNI_A: LibxsmmBitfield = 2048;
pub const LIBXSMM_GEMM_FLAG_VNNI_B: LibxsmmBitfield = 4096;
pub const LIBXSMM_GEMM_FLAG_A_UNSIGNED: LibxsmmBitfield = 256;

// ============================================================================
// Architecture IDs (from libxsmm_cpuid.h)
// ============================================================================

pub const LIBXSMM_TARGET_ARCH_AVX2: c_int = 1006;
pub const LIBXSMM_TARGET_ARCH_AVX512_SKX: c_int = 1101;
pub const LIBXSMM_TARGET_ARCH_AVX512_CLX: c_int = 1102; // VNNI
pub const LIBXSMM_TARGET_ARCH_AVX512_CPX: c_int = 1103; // BF16
pub const LIBXSMM_TARGET_ARCH_AVX512_SPR: c_int = 1104; // AMX

// ============================================================================
// Struct types (from libxsmm_typedefs.h)
// ============================================================================

/// GEMM shape descriptor — kernel dimensions + data types.
/// From libxsmm_typedefs.h line 734.
#[repr(C)]
#[derive(Clone)]
pub struct LibxsmmGemmShape {
    pub m: LibxsmmBlasint,
    pub n: LibxsmmBlasint,
    pub k: LibxsmmBlasint,
    pub lda: LibxsmmBlasint,
    pub ldb: LibxsmmBlasint,
    pub ldc: LibxsmmBlasint,
    pub a_in_type: c_int,
    pub b_in_type: c_int,
    pub out_type: c_int,
    pub comp_type: c_int,
}

/// Data carrier for GEMM operands. Only `primary` matters for basic GEMM.
/// From libxsmm_typedefs.h line 577.
#[repr(C)]
pub struct LibxsmmMatrixArg {
    pub primary: *const c_void,
    pub secondary: *const c_void,
    pub tertiary: *const c_void,
    pub quaternary: *const c_void,
    pub quinary: *const c_void,
    pub senary: *const c_void,
}

impl LibxsmmMatrixArg {
    pub fn from_ptr(ptr: *const c_void) -> Self {
        Self {
            primary: ptr,
            secondary: std::ptr::null(),
            tertiary: std::ptr::null(),
            quaternary: std::ptr::null(),
            quinary: std::ptr::null(),
            senary: std::ptr::null(),
        }
    }
}

/// Operator state for GEMM. Zeroed for basic GEMM.
/// From libxsmm_typedefs.h line 586.
#[repr(C)]
pub struct LibxsmmMatrixOpArg {
    pub primary: *const c_void,
    pub secondary: *const c_void,
    pub tertiary: *const c_void,
    pub quaternary: *const c_void,
}

impl Default for LibxsmmMatrixOpArg {
    fn default() -> Self {
        Self {
            primary: std::ptr::null(),
            secondary: std::ptr::null(),
            tertiary: std::ptr::null(),
            quaternary: std::ptr::null(),
        }
    }
}

/// Call-site argument bundle for JIT kernels.
/// From libxsmm_typedefs.h line 716.
#[repr(C)]
pub struct LibxsmmGemmParam {
    pub op: LibxsmmMatrixOpArg,
    pub a: LibxsmmMatrixArg,
    pub b: LibxsmmMatrixArg,
    pub c: LibxsmmMatrixArg,
}

/// JIT-compiled GEMM function pointer type.
pub type LibxsmmGemmFunction = unsafe extern "C" fn(*const LibxsmmGemmParam);

// ============================================================================
// FFI function bindings
// ============================================================================

#[link(name = "xsmm")]
extern "C" {
    // Lifecycle
    pub fn libxsmm_init();
    pub fn libxsmm_finalize();

    // Architecture detection
    pub fn libxsmm_get_target_archid() -> c_int;

    // Shape constructor (convenience — fills a struct)
    pub fn libxsmm_create_gemm_shape(
        m: LibxsmmBlasint,
        n: LibxsmmBlasint,
        k: LibxsmmBlasint,
        lda: LibxsmmBlasint,
        ldb: LibxsmmBlasint,
        ldc: LibxsmmBlasint,
        a_in_type: c_int,
        b_in_type: c_int,
        out_type: c_int,
        comp_type: c_int,
    ) -> LibxsmmGemmShape;

    // JIT dispatch — returns function pointer to generated machine code.
    // Returns null if shape/type unsupported for this CPU.
    pub fn libxsmm_dispatch_gemm(
        gemm_shape: LibxsmmGemmShape,
        gemm_flags: LibxsmmBitfield,
        prefetch_flags: LibxsmmBitfield,
    ) -> Option<LibxsmmGemmFunction>;

    // BLAS-compatible SGEMM (auto-JIT internally, fallback path)
    pub fn libxsmm_sgemm(
        transa: *const c_char,
        transb: *const c_char,
        m: *const LibxsmmBlasint,
        n: *const LibxsmmBlasint,
        k: *const LibxsmmBlasint,
        alpha: *const c_float,
        a: *const c_float,
        lda: *const LibxsmmBlasint,
        b: *const c_float,
        ldb: *const LibxsmmBlasint,
        beta: *const c_float,
        c: *mut c_float,
        ldc: *const LibxsmmBlasint,
    );
}

// ============================================================================
// Safe wrappers
// ============================================================================

/// Safe wrapper for SGEMM via LIBXSMM.
pub unsafe fn xsmm_sgemm(
    transa: u8,
    transb: u8,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    b: *const f32,
    ldb: i32,
    beta: f32,
    c: *mut f32,
    ldc: i32,
) {
    let transa_char = transa as c_char;
    let transb_char = transb as c_char;
    let m_blasint = m as LibxsmmBlasint;
    let n_blasint = n as LibxsmmBlasint;
    let k_blasint = k as LibxsmmBlasint;
    let lda_blasint = lda as LibxsmmBlasint;
    let ldb_blasint = ldb as LibxsmmBlasint;
    let ldc_blasint = ldc as LibxsmmBlasint;

    libxsmm_sgemm(
        &transa_char,
        &transb_char,
        &m_blasint,
        &n_blasint,
        &k_blasint,
        &alpha,
        a,
        &lda_blasint,
        b,
        &ldb_blasint,
        &beta,
        c,
        &ldc_blasint,
    );
}

/// Cached JIT kernel for a fixed GEMM shape.
/// Dispatch cost paid once; hot-path is a single indirect call.
pub struct JitKernel {
    kernel: LibxsmmGemmFunction,
}

impl JitKernel {
    /// Try to dispatch a JIT kernel for f32 GEMM.
    /// Returns None if LIBXSMM can't JIT for this shape.
    pub fn f32_gemm(m: i32, n: i32, k: i32) -> Option<Self> {
        unsafe {
            libxsmm_init();
            let shape = libxsmm_create_gemm_shape(
                m,
                n,
                k,
                m,  // lda
                k,  // ldb
                m,  // ldc
                LIBXSMM_DATATYPE_F32,
                LIBXSMM_DATATYPE_F32,
                LIBXSMM_DATATYPE_F32,
                LIBXSMM_DATATYPE_F32,
            );
            let kernel = libxsmm_dispatch_gemm(shape, LIBXSMM_GEMM_FLAG_BETA_0, 0)?;
            Some(Self { kernel })
        }
    }

    /// Try to dispatch a JIT kernel for BF16→f32 GEMM.
    /// Requires CPX+ (VDPBF16PS) or SPR+ (AMX TDPBF16PS).
    pub fn bf16_gemm(m: i32, n: i32, k: i32) -> Option<Self> {
        unsafe {
            libxsmm_init();
            let shape = libxsmm_create_gemm_shape(
                m,
                n,
                k,
                m,
                k,
                m,
                LIBXSMM_DATATYPE_BF16,
                LIBXSMM_DATATYPE_BF16,
                LIBXSMM_DATATYPE_F32,
                LIBXSMM_DATATYPE_F32,
            );
            let kernel = libxsmm_dispatch_gemm(shape, LIBXSMM_GEMM_FLAG_BETA_0, 0)?;
            Some(Self { kernel })
        }
    }

    /// Call the JIT kernel. a/b/c must be valid for the dispatched shape.
    pub unsafe fn call(
        &self,
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
    ) {
        let param = LibxsmmGemmParam {
            op: LibxsmmMatrixOpArg::default(),
            a: LibxsmmMatrixArg::from_ptr(a),
            b: LibxsmmMatrixArg::from_ptr(b),
            c: LibxsmmMatrixArg::from_ptr(c as *const c_void),
        };
        (self.kernel)(&param);
    }
}
