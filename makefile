AS = nasm
LD = gcc

ASFLAGS = -f win64
LDFLAGS = -m64

TARGET = $(BINDIR)/mlp_x64.exe

SRCDIR = src
TESTDIR = testing
BINDIR = bin
OBJDIR = obj

SRCS = $(wildcard $(SRCDIR)/*.asm) $(wildcard $(TESTDIR)/*.asm)

OBJS = $(patsubst $(SRCDIR)/%.asm,$(OBJDIR)/%.obj,$(wildcard $(SRCDIR)/*.asm)) \
       $(patsubst $(TESTDIR)/%.asm,$(OBJDIR)/%.obj,$(wildcard $(TESTDIR)/*.asm))

all: $(BINDIR) $(OBJDIR) $(TARGET)

$(BINDIR):
	@if not exist $(BINDIR) mkdir $(BINDIR)

$(OBJDIR):
	@if not exist $(OBJDIR) mkdir $(OBJDIR)

$(TARGET): $(OBJS)
	$(LD) $(LDFLAGS) -Wl,-e,main -o $@ $^

$(OBJDIR)/%.obj: $(SRCDIR)/%.asm
	$(AS) $(ASFLAGS) $< -o $@

$(OBJDIR)/%.obj: $(TESTDIR)/%.asm
	$(AS) $(ASFLAGS) $< -o $@

clean:
	@if exist $(OBJDIR)\*.obj del /f $(OBJDIR)\*.obj
	@if exist $(BINDIR)\*.exe del /f $(BINDIR)\*.exe

run: $(TARGET)
	.\$(TARGET)

.PHONY: all clean run
