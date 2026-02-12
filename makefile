AS = nasm
LD = gcc

ASFLAGS = -f win64
LDFLAGS = -m64

TARGET = $(BINDIR)/mlp-x64.exe

SRCDIR = src
SRC_LIBS = $(SRCDIR)/libs
SRC_MLP = $(SRCDIR)/mlp
TESTDIR = testing
BINDIR = bin
OBJDIR = obj

SRCS = $(wildcard $(SRCDIR)/*.asm) \
       $(wildcard $(SRC_LIBS)/*.asm) \
       $(wildcard $(SRC_MLP)/*.asm) \
       $(wildcard $(TESTDIR)/*.asm)

OBJS = $(patsubst $(SRCDIR)/%.asm,$(OBJDIR)/%.obj,$(wildcard $(SRCDIR)/*.asm)) \
       $(patsubst $(SRC_LIBS)/%.asm,$(OBJDIR)/libs_%.obj,$(wildcard $(SRC_LIBS)/*.asm)) \
       $(patsubst $(SRC_MLP)/%.asm,$(OBJDIR)/mlp_%.obj,$(wildcard $(SRC_MLP)/*.asm)) \
       $(patsubst $(TESTDIR)/%.asm,$(OBJDIR)/%.obj,$(wildcard $(TESTDIR)/*.asm))

all: $(BINDIR) $(OBJDIR) $(TARGET)

$(BINDIR):
	@if not exist $(BINDIR) mkdir $(BINDIR)

$(OBJDIR):
	@if not exist $(OBJDIR) mkdir $(OBJDIR)

$(TARGET): $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^

$(OBJDIR)/%.obj: $(SRCDIR)/%.asm
	$(AS) $(ASFLAGS) $< -o $@

$(OBJDIR)/libs_%.obj: $(SRC_LIBS)/%.asm
	$(AS) $(ASFLAGS) $< -o $@

$(OBJDIR)/mlp_%.obj: $(SRC_MLP)/%.asm
	$(AS) $(ASFLAGS) $< -o $@

$(OBJDIR)/%.obj: $(TESTDIR)/%.asm
	$(AS) $(ASFLAGS) $< -o $@

clean:
	@if exist $(OBJDIR)\*.obj del /f $(OBJDIR)\*.obj
	@if exist $(BINDIR)\*.exe del /f $(BINDIR)\*.exe

run: $(TARGET)
	.\$(TARGET)

.PHONY: all clean run
