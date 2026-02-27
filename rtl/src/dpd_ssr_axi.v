//==============================================================================
// DPD SSR AXI Wrapper for ZCU104
// Interface: AXI4-Stream I/O + AXI4-Lite control
//==============================================================================

module dpd_ssr_axi (
    // AXI4-Lite Control Interface
    input  wire        S00_AXI_aclk,
    input  wire        S00_AXI_aresetn,
    input  wire [5:0]  S00_AXI_awaddr,
    input  wire [2:0]  S00_AXI_awprot,
    input  wire        S00_AXI_awvalid,
    output reg         S00_AXI_awready,
    input  wire [31:0] S00_AXI_wdata,
    input  wire [3:0]  S00_AXI_wstrb,
    input  wire        S00_AXI_wvalid,
    output reg         S00_AXI_wready,
    output reg  [1:0]  S00_AXI_bresp,
    output reg         S00_AXI_bvalid,
    input  wire        S00_AXI_bready,
    input  wire [5:0]  S00_AXI_araddr,
    input  wire [2:0]  S00_AXI_arprot,
    input  wire        S00_AXI_arvalid,
    output reg         S00_AXI_arready,
    output reg  [31:0] S00_AXI_rdata,
    output reg  [1:0]  S00_AXI_rresp,
    output reg         S00_AXI_rvalid,
    input  wire        S00_AXI_rready,

    // AXI4-Stream Input (32-bit I/Q)
    input  wire        S00_AXIS_aclk,
    input  wire        S00_AXIS_aresetn,
    input  wire [31:0] S00_AXIS_tdata,
    input  wire [3:0]  S00_AXIS_tkeep,
    input  wire        S00_AXIS_tlast,
    input  wire        S00_AXIS_tvalid,
    output wire        S00_AXIS_tready,

    // AXI4-Stream Output (160-bit: 5x I/Q)
    output wire [159:0] M00_AXIS_tdata,
    output wire [19:0]  M00_AXIS_tkeep,
    output wire         M00_AXIS_tlast,
    output wire         M00_AXIS_tvalid,
    input  wire         M00_AXIS_tready,

    // System signals
    input  wire clk,      // 200 MHz processing clock
    input  wire rst_n
);

    // Register file
    reg  [31:0] ctrl_reg;    // Control register
    reg  [31:0] status_reg;  // Status register
    reg  [31:0] temp_bank;   // Temperature compensation bank select
    wire [31:0] version_reg = 32'h01000000;  // v1.0.0

    // Input/output signals
    wire [31:0] din_i, din_q;
    wire        din_valid, din_ready;
    wire [159:0] dout_data;
    wire [4:0]  dout_i_0, dout_i_1, dout_i_2, dout_i_3, dout_i_4;
    wire [4:0]  dout_q_0, dout_q_1, dout_q_2, dout_q_3, dout_q_4;
    wire        dout_valid, core_ready;

    // FIFO for decoupling domains (AXI clock to processing clock)
    wire [31:0] fifo_din, fifo_dout;
    wire        fifo_wr_en, fifo_rd_en, fifo_full, fifo_empty;
    wire [11:0] fifo_level;

    // Core instantiation
    dpd_top_ssr dpd_core (
        .clk(clk),
        .rst_n(rst_n),
        .din_i(din_i),
        .din_q(din_q),
        .din_valid(din_valid),
        .din_ready(core_ready),
        .temp_bank(temp_bank[1:0]),
        .dout_i_0(dout_i_0), .dout_q_0(dout_q_0),
        .dout_i_1(dout_i_1), .dout_q_1(dout_q_1),
        .dout_i_2(dout_i_2), .dout_q_2(dout_q_2),
        .dout_i_3(dout_i_3), .dout_q_3(dout_q_3),
        .dout_i_4(dout_i_4), .dout_q_4(dout_q_4),
        .dout_valid(dout_valid)
    );

    // Input path: Unpack 32-bit AXI data into I/Q
    assign din_i = S00_AXIS_tdata[15:0];
    assign din_q = S00_AXIS_tdata[31:16];
    assign din_valid = S00_AXIS_tvalid && ~fifo_full;
    assign S00_AXIS_tready = ~fifo_full && core_ready;

    // Output path: Pack 5x I/Q into 160-bit AXI
    assign M00_AXIS_tdata = {
        dout_q_4, dout_i_4,
        dout_q_3, dout_i_3,
        dout_q_2, dout_i_2,
        dout_q_1, dout_i_1,
        dout_q_0, dout_i_0
    };
    assign M00_AXIS_tkeep = 20'hFFFFF;
    assign M00_AXIS_tlast = 1'b0;
    assign M00_AXIS_tvalid = dout_valid;

    //==========================================================================
    // AXI4-Lite Control Interface
    //==========================================================================

    always @(posedge S00_AXI_aclk or negedge S00_AXI_aresetn) begin
        if (~S00_AXI_aresetn) begin
            ctrl_reg     <= 32'h0;
            temp_bank    <= 32'h0;
            status_reg   <= 32'h0;
            S00_AXI_awready <= 1'b0;
            S00_AXI_wready  <= 1'b0;
            S00_AXI_bvalid  <= 1'b0;
            S00_AXI_arready <= 1'b0;
            S00_AXI_rvalid  <= 1'b0;
        end else begin
            // Write address
            if (S00_AXI_awvalid && ~S00_AXI_awready) begin
                S00_AXI_awready <= 1'b1;
            end else begin
                S00_AXI_awready <= 1'b0;
            end

            // Write data
            if (S00_AXI_wvalid && ~S00_AXI_wready) begin
                S00_AXI_wready <= 1'b1;
                case (S00_AXI_awaddr[5:2])
                    4'h0: ctrl_reg  <= S00_AXI_wdata;
                    4'h2: temp_bank <= S00_AXI_wdata;
                endcase
            end else begin
                S00_AXI_wready <= 1'b0;
            end

            // Write response
            if (S00_AXI_bready) begin
                S00_AXI_bvalid <= 1'b0;
            end else if (S00_AXI_wvalid && S00_AXI_wready) begin
                S00_AXI_bvalid <= 1'b1;
                S00_AXI_bresp <= 2'b00;  // OKAY
            end

            // Read address
            if (S00_AXI_arvalid && ~S00_AXI_arready) begin
                S00_AXI_arready <= 1'b1;
            end else begin
                S00_AXI_arready <= 1'b0;
            end

            // Read data
            if (S00_AXI_arvalid && S00_AXI_arready) begin
                S00_AXI_rvalid <= 1'b1;
                case (S00_AXI_araddr[5:2])
                    4'h0: S00_AXI_rdata <= ctrl_reg;
                    4'h1: S00_AXI_rdata <= status_reg;
                    4'h2: S00_AXI_rdata <= temp_bank;
                    4'h3: S00_AXI_rdata <= version_reg;
                    default: S00_AXI_rdata <= 32'h0;
                endcase
                S00_AXI_rresp <= 2'b00;  // OKAY
            end else if (S00_AXI_rready) begin
                S00_AXI_rvalid <= 1'b0;
            end
        end
    end

    // Status register updates
    always @(posedge S00_AXI_aclk) begin
        status_reg[0] <= dout_valid;     // BUSY
        status_reg[1] <= ~fifo_full;     // READY
        status_reg[11:2] <= fifo_level;  // FIFO level
    end

endmodule
